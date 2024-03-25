from typing import Dict, Tuple, Optional
import timeit
import logging
from pathlib import Path
import json
import pickle


import box
import yaml
import json
import typer
from tqdm import tqdm
import pandas as pd
import numpy as np
from gluonts.core.serde import dump_json
from gluonts.dataset.split import split, TestData, TrainingDataset
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import (
    SMAPE,
    MASE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss,
    AverageMeanScaledQuantileLoss,
    MAECoverage,
)
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.common import FileDataset
from gluonts.model.forecast import SampleForecast

from mlfdatasets import RemoteDataset
from mlfmodels import PretrainedModel

app = typer.Typer(pretty_exceptions_enable=False)
METRICS = [
    SMAPE(),
    MASE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]
    ),
    AverageMeanScaledQuantileLoss(
        quantile_levels=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]
    ),
    MAECoverage(
        quantile_levels=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]
    ),
]


def get_dataset_by_config(
    dataset_config: Dict, compute_metrics: bool = True
) -> Tuple[TrainingDataset, TestData]:
    """Get dataset from a dataset config. The dataset config must be a dictionary
    with the following keys.
        source: one of {gluonts, mlfdatasets, file}
        if the source is gluonts, the following keys are supported:
            name: name of the dataset (required)
        if the source is mlfdatasets, the following keys are supported:
            name: name of the dataset (required)
            prediction_length: the prediction length (required)
            subsets: list of subsets to load (optional)
            num_rolls: num of rolling evals (optional, default = 1)
            offset: negative train/test split index (optional)
            date: split forecast start date (optional)
            info_fields: extra info fields in the data (optional)
        if the source is file, the following keys are supported:
            path: path to the dataset (required)
            freq: frequency (required)
            prediction_length: the prediction length (required)
            num_rolls: num of rolling evals (optional, default = 1)
            offset: negative train/test split index (optional)
            date: split forecast start date (optional)

    Parameters
    ----------
    dataset_config
        A dataset config dict as described above.

    Returns
    -------
        (training_dataset, test_data)
    """

    def get_split_info():
        prediction_length = dataset_config["prediction_length"]
        num_rolls = dataset_config.get("num_rolls", 1)
        offset = dataset_config.get("offset")
        date = dataset_config.get("date")
        assert not (offset and date), "Both offset and date cannot be in the config"
        if date is not None:
            date = pd.Period(date, freq=freq) - 1
        if offset is None and date is None:
            offset = -prediction_length * num_rolls
        return offset, date, prediction_length, num_rolls

    source = dataset_config["source"]

    if source == "gluonts":
        assert compute_metrics
        dataset_name = dataset_config["name"]
        dataset = get_dataset(dataset_name)
        train_dataset = dataset.train
        prediction_length = dataset.metadata.prediction_length
        _, test_template = split(dataset.test, offset=-prediction_length)
        test_data = test_template.generate_instances(prediction_length)

        return train_dataset, test_data, prediction_length
    elif source == "mlfdatasets":
        dataset_name = dataset_config["name"]
        subsets = dataset_config.get("subsets", None)
        dataset = (
            RemoteDataset.from_dataset_name(dataset_name)
            .load(subsets=subsets)
            .to_gluonts(info_fields=dataset_config.get("info_fields"))
        )
        freq = dataset.dataset.remote.frequency
        offset, date, prediction_length, num_rolls = get_split_info()
        train_dataset, test_template = split(dataset, offset=offset, date=date)
        test_data = test_template.generate_instances(
            prediction_length if compute_metrics else 0,
            windows=num_rolls,
        )
        return train_dataset, test_data, prediction_length
    elif source == "file":
        path = dataset_config["path"]
        freq = dataset_config["freq"]
        dataset = FileDataset(path=path, freq=freq)
        offset, date, prediction_length, num_rolls = get_split_info()
        train_dataset, test_template = split(dataset, offset=offset, date=date)
        test_data = test_template.generate_instances(
            prediction_length if compute_metrics else 0,
            windows=num_rolls,
        )

        return train_dataset, test_data, prediction_length
    else:
        raise ValueError(f"Unknown dataset source: {source}")


class TimeOutError(Exception):
    pass


class NanForecastError(Exception):
    pass


DEFAULT_TIMEOUT_SECONDS = 6 * 60 * 60  # six hours


def remove_prefix(name):
    if "/" in name:
        ind = name.find("/")
        return name[ind + 1 :]
    else:
        return name


def evaluate_dataset(
    dataset_name,
    test_data: TestData,
    pred_folder,
    compute_metrics=True,
):
    dataset_name = remove_prefix(dataset_name)

    with open(f"{pred_folder}/{dataset_name}.json", "r") as f:
        predictions = json.load(f)

    error_sample_count = 0
    total_sample_count = 0
    for i, pred in enumerate(predictions):
        valid_samples = []
        total_sample_count += len(pred["predictions"])
        for j, sample in enumerate(pred["predictions"]):
            if abs(sample[-1]) < min(abs(np.array(sample[:-1]))) / 10:
                print(f"{dataset_name} {i}th time series, {j}th sample")
                print("Predictions", sample)
                error_sample_count += 1
            else:
                valid_samples.append(sample)
                # print("Completion_list", pred["completions_list"][0][j])
        pred["predictions"] = valid_samples

    # verify the groundtruth values are the same
    assert len(predictions) == len(test_data)
    # for pred, output in zip(predictions, test_data.label):
    #    np.isclose(np.array(pred["targets"]).shape, output["target"].shape)
    #    assert len(pred["predictions"]) >= 19, len(pred["predictions"])

    forecasts = [
        SampleForecast(
            samples=np.array(pred["predictions"]), start_date=output["start"]
        )
        for pred, output in zip(predictions, test_data.label)
    ]

    agg_metrics = item_metrics = None
    start_time = timeit.default_timer()
    if compute_metrics:
        agg_metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=METRICS,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        item_metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=METRICS,
                axis=1,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

    end_time = timeit.default_timer()
    print(
        f"*** {dataset_name} Total predictions {total_sample_count}, error predictions {error_sample_count}"
    )
    return forecasts, {
        "agg_metrics": agg_metrics,
        "item_metrics": item_metrics,
        "inference_time": sum([pred.get("inference_time", 0) for pred in predictions]),
        "metric_computation_time": end_time - start_time,
    }


def save_results_to_file(results, path):
    with open(path, "w") as fp:
        json.dump(results, fp)


def save_forecasts_to_file(forecasts, path):
    with open(path, "w") as fp:
        fp.write(dump_json(forecasts))


def read_their_data(dataset_name):
    pkl_path = f"llmtime/datasets/monash/{dataset_name}.pkl"
    if Path(pkl_path).exists():
        data = pickle.load(open(pkl_path, "rb"))[0]
        return data
    else:
        return None


def main(
    pred_folder,
    config: Path = Path("config.yaml"),
    out_dir: Path = Path("./results/"),
    job_id: str = "local",
    instance_type: str = "local",
    save_forecasts: bool = False,
    compute_metrics: bool = True,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(config) as fp:
        eval_config = box.Box(yaml.safe_load(fp))

    backtest_configs = eval_config.backtests
    backtest_num = 1
    model = "llmtime"
    for backtest_config in backtest_configs:
        logger.info(f"Loading dataset: {backtest_config}")
        _, test_data, prediction_length = get_dataset_by_config(
            backtest_config, compute_metrics
        )
        dataset_name = remove_prefix(backtest_config["name"])
        pred_path = f"{pred_folder}/{dataset_name}.json"
        if not Path(pred_path).exists():
            continue

        freq = test_data.dataset.dataset.remote.frequency
        try:
            forecasts, results = evaluate_dataset(
                backtest_config["name"],
                test_data,
                pred_folder,
                compute_metrics=compute_metrics,
            )
            logger.info("Saving results")
            save_results_to_file(
                {
                    "job_id": job_id,
                    "instance_type": instance_type,
                    "model_config": {"prefix": model},
                    "backtest_config": backtest_config,
                    "num_samples": 20,
                    **results,
                },
                out_dir / f"backtest_{backtest_num}.json",
            )
            if save_forecasts:
                save_forecasts_to_file(
                    forecasts,
                    out_dir / f"forecasts_{backtest_num}.json",
                )
        except (TimeOutError, NanForecastError, ValueError) as e:
            logger.info(f"Skipping backtest: {e}")

        backtest_num += 1


def check_predictions():
    config: Path = Path("llmtime/config.yaml")
    with open(config) as fp:
        eval_config = box.Box(yaml.safe_load(fp))

    def has_problem(dataset_name):
        json_path = f"llmtime/predictions_20_sm1.4/{dataset_name}.json"
        with open(json_path, "r") as f:
            preds = json.load(f)
        for i, pred in enumerate(preds):
            for j, sample in enumerate(pred["predictions"]):
                if abs(sample[-1]) < min(abs(np.array(sample[:-1]))) / 10:
                    print(f"{dataset_name} {i}th time series, {j}th sample")
                    print("Predictions", sample)
                    print("Completion_list", pred["completions_list"][0][j])
        return False

    backtest_configs = eval_config.backtests
    for backtest_config in backtest_configs:
        dataset_name = backtest_config["name"]
        dataset_name = remove_prefix(dataset_name)

        json_path = f"llmtime/predictions_20_sm1.4/{dataset_name}.json"
        if not Path(json_path).exists():
            continue
        if has_problem(dataset_name):
            print(dataset_name, "has problem")
        else:
            pass
            print(dataset_name, "is fine")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("mlfevals")
    logger.setLevel(logging.INFO)
    main(pred_folder="predictions_20_llmtime", out_dir=Path("results_llmtime"))
    main(pred_folder="predictions_20_sm1.4_subset", out_dir=Path("results_ours"))
    # main(pred_folder="predictions_20_sm1.4", out_dir=Path("results_sm1.4"))
    # read_their_data("cif_2016")
    # check_predictions()
