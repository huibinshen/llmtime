import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_dataset_name(file_path):
    file_path = str(file_path)
    start_ind = file_path.find("llama2_70B") + len("llama2_70B/")
    dataset_with_ind = file_path[start_ind:].split("/")[0]
    ind = dataset_with_ind.split("_")[-1]
    dataset_name_end = len(ind) + 1
    return dataset_with_ind[:-dataset_name_end], ind


def load_llmtime_prediction(dataset):
    results = []
    for pickle_file in Path("precomputed_outputs/llama_2_monash/llama2_70B").glob(
        f"{dataset}*/*.pkl"
    ):
        assert pickle_file.exists()
        dataset, ind = find_dataset_name(pickle_file)
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        samples = data["samples"]
        assert isinstance(samples, pd.DataFrame), type(samples)
        predictions = samples.to_numpy()
        results.append({"ind": int(ind), "predictions": predictions})
    return results


def load_our_predictions(dataset):
    # Use "predictions_20" for default STEP_MULTIPLIER
    # use "predictions_20_sm1.4" for STEP_MULTIPLIER=1.4
    with open(f"predictions_20_sm1.4/{dataset}.json", "r") as f:
        data = json.load(f)
    return data


# common monash datasets with same prediction length
datasets = [
    # "covid_deaths",
    "nn5_weekly",
    "cif_2016",
    "hospital",
    #    "fred_md",
    #    "tourism_monthly",
    "tourism_yearly",
    "tourism_quarterly",
]


def plot():
    def _find_by_ind(predictions, ind):
        for p in predictions:
            if p["ind"] == ind:
                return p["predictions"]
        return None

    n_ts_plot = 5
    for dataset in datasets:

        our_predictions = load_our_predictions(dataset)
        llmtime_predictions = load_llmtime_prediction(dataset)

        fig, axes = plt.subplots(n_ts_plot, 2, sharey="row")
        row_ind = 0
        for ts_ind, our in enumerate(our_predictions):
            if row_ind > n_ts_plot - 1:
                break
            ind = our["ind"]
            our_pred = our["predictions"]
            their_pred = _find_by_ind(llmtime_predictions, ind)
            assert their_pred is not None
            for sample_ind, (op, tp) in enumerate(zip(our_pred, their_pred)):
                print(ts_ind, sample_ind)
                print("completions_list", our["completions_list"][0][sample_ind])
                print("ours", op[-5:])
                print("theirs", tp[-5:])

            our_mean = np.mean(our_pred, axis=0)
            our_std = 2.0 * np.std(our_pred, axis=0)
            their_mean = np.mean(their_pred, axis=0)
            their_std = 2.0 * np.std(their_pred, axis=0)
            axes[row_ind, 0].scatter(range(len(our_mean)), our_mean, label="Ours")
            axes[row_ind, 1].scatter(
                range(len(their_mean)), their_mean, label="LLMTime"
            )
            axes[row_ind, 0].errorbar(range(len(our_mean)), our_mean, our_std)
            axes[row_ind, 1].errorbar(range(len(their_mean)), their_mean, their_std)
            if row_ind == 0:
                axes[row_ind, 0].set_title(f"{dataset}")
                axes[row_ind, 1].set_title(f"{dataset}")
                axes[row_ind, 0].legend()
                axes[row_ind, 1].legend()
            row_ind += 1
        plt.tight_layout()
        plt.savefig(f"compare_sm1.4/{dataset}.pdf")


if __name__ == "__main__":
    plot()
