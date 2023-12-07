from pathlib import Path
import json

import numpy as np
import pandas as pd
from time import time
import yaml
from tqdm import tqdm

from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data, truncate_input, get_scaler
from models.validation_likelihood_tuning import get_autotuned_predictions_data

import mlfdatasets
from mlfdatasets import RemoteDataset

with open("bin/zero-shot.yaml", "r") as file:
    configs = yaml.safe_load(file)


def cap():
    res = []
    for i, config in enumerate(configs):
        #if not config["name"] == "monash/hospital":
        #    continue
        dataset_name = config["name"]
        # dataset_name = remove_prefix(config["name"])
        try:
            remote_dataset = RemoteDataset.from_dataset_name(dataset_name)
        except Exception as e:
            print(e)
            continue
        pred_length = config["prediction_length"]
        dataset = remote_dataset.load()
        res.append({"name": dataset_name, "n_univarite": len(dataset), "pred_lenght": pred_length, "n_obs": len(dataset)*pred_length})
        print(i, dataset_name, len(dataset), pred_length)
    df = pd.DataFrame(res).sort_values("n_obs")
    print(df)


if __name__ == "__main__":
    cap()