from functools import partial

import pandas as pd
import torch
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC,
                                         BinaryF1Score, BinaryAveragePrecision, BinaryConfusionMatrix)
from evaluate.inference import ColumnNames
from pathlib import Path
from evaluate.confusion_matrix import run_confusion_matrix
import json


def run_metrics(y_prob: torch.Tensor,
                y_true: torch.Tensor,
                threshold: float = 0.5):
    metrics = {
        "accuracy": BinaryAccuracy(threshold=threshold),
        "precision": BinaryPrecision(threshold=threshold),
        "recall": BinaryRecall(threshold=threshold),
        "f1": BinaryF1Score(threshold=threshold),
        "auroc": BinaryAUROC(),
        "average_precision": BinaryAveragePrecision(),
        "confusion_matrix": partial(run_confusion_matrix, threshold=threshold, return_type='dict')
    }

    results = {}

    for name, metric in metrics.items():
        value = metric(y_prob, y_true)

        if isinstance(value, torch.Tensor):
            value = value.item()

        results[name] = value

    return results


def run_metrics_pd(df: pd.DataFrame,
                   threshold: float = 0.5,
                   save_results: bool = True,
                   save_directory: str | Path | None = None,
                   file_name: str = None):
    if save_results:
        if save_directory is None:
            raise ValueError("save_directory not specified")
        else:
            save_directory: Path = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = "metrics.json"
    y_prob = torch.tensor(df[ColumnNames.predictions].values, dtype=torch.float)
    y_true = torch.tensor(df[ColumnNames.labels].values, dtype=torch.int)

    results = run_metrics(y_prob, y_true, threshold=threshold)
    if save_results:
        with open(save_directory / file_name, "w") as f:
            json.dump(results, f, indent=4)
    return results


def run_metrics_csv(csv_path: str | Path,
                    threshold: float = 0.5,
                    save_results: bool = True,
                    save_directory: str | Path = None,
                    file_name: str = None):
    # make sure that we can load csv
    csv_path: Path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # resolve save directory
    if save_results:
        if save_directory is None:
            save_directory = csv_path.parent
        else:
            save_directory: Path = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)

    # run metrics
    return run_metrics_pd(df, threshold=threshold, save_results=save_results, save_directory=save_directory, file_name=file_name)
