from pathlib import Path
import json

import numpy as np
import pandas as pd
from time import time
import yaml
from tqdm import tqdm

from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data
from models.validation_likelihood_tuning import get_autotuned_predictions_data

import mlfdatasets
from mlfdatasets import RemoteDataset


df = mlfdatasets.summarize()
df = df.reset_index()
datasets_names = df["prefix"].to_list()


model_predict_fns = {
    "text-davinci-003": get_llmtime_predictions_data,
    "llama-7b": get_llmtime_predictions_data,
    "llama-70b": get_llmtime_predictions_data,
}


def remove_prefix(name):
    if "/" in name:
        ind = name.find("/")
        return name[ind + 1 :]
    else:
        return name


with open("bin/zero-shot-small.yaml", "r") as file:
    configs = yaml.safe_load(file)["backtests"]


llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(
        base=10,
        prec=3,
        time_sep=",",
        bit_sep="",
        plus_sign="",
        minus_sign="-",
        signed=True,
    ),
)
llama_hypers.update({"model": "llama-70b"})
num_samples = 20
context_length = 512


def sort_configs_by_num_obs(configs):
    for config in configs:
        dataset_name = config["name"]
        # dataset_name = remove_prefix(config["name"])
        try:
            remote_dataset = RemoteDataset.from_dataset_name(dataset_name)
        except Exception as e:
            print(e)
            continue
        pred_length = config["prediction_length"]
        dataset = remote_dataset.load()
        config["num_obs"] =  len(dataset) * pred_length
    return sorted(configs, key=lambda x: x["num_obs"])

configs = sort_configs_by_num_obs(configs)

def estimate():
    for i, config in enumerate(configs):
        print(i, config)
        dataset_name = config["name"]

        try:
            remote_dataset = RemoteDataset.from_dataset_name(dataset_name)
        except Exception as e:
            continue

        json_path = (
            Path(__file__).parent / f"predictions_20_sm1.4/{remove_prefix(dataset_name)}.json"
        )

        if json_path.exists():
            print(json_path, "exists")
            continue

        dataset = remote_dataset.load()
        gluonts_dataset = dataset.to_gluonts()

        pred_length = config["prediction_length"]

        train_list = [
            ts["target"][config["offset"] - context_length : config["offset"]]
            for i, ts in enumerate(gluonts_dataset)
        ]

        if config["offset"] + pred_length < 0:
            test_list = [
                ts["target"][config["offset"] : config["offset"] + pred_length]
                for i, ts in enumerate(gluonts_dataset)
            ]
        else:
            test_list = [
                ts["target"][config["offset"] :] for i, ts in enumerate(gluonts_dataset)
            ]

        # for hospital dataset, json can't save int64
        if test_list[0].dtype == np.int64:
            new_test_list = []
            for ts in test_list:
                ts = list(map(int, ts))
                new_test_list.append(ts)
            test_list = new_test_list

        assert len(train_list) == len(test_list), (len(train_list), len(test_list))

        result = []
        for i in tqdm(range(len(train_list))):
            print(i, len(train_list), len(train_list[i]), len(test_list[i]))
            start_time = time()
            preds = get_autotuned_predictions_data(
                train_list[i],
                test_list[i],
                llama_hypers,
                num_samples,
                model_predict_fns[llama_hypers["model"]],
                verbose=False,
                parallel=False,
            )
            end_time = time()
            medians = preds["median"]
            targets = np.array(test_list[i])
            mae = np.mean(np.abs(medians - targets))

            d = {
                "ind": i,
                "predictions": [list(arr) for arr in preds["samples"].values],
                "targets": list(test_list[i]),
                "mae": mae,
                "pred_length": pred_length,
                "inference_time": end_time - start_time,
            }
            result.append(d)

        with open(json_path, "w") as f:
            json.dump(result, f)
        dataset.purge()

    
if __name__ == "__main__":
    estimate()
