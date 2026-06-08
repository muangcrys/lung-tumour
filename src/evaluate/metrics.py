import pandas as pd
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryF1Score
from evaluate.inference import ColumnNames
from pathlib import Path
import json


def run_metrics(y_prob: torch.Tensor,
                y_true: torch.Tensor,
                threshold: float = 0.5):
    metrics = {
        "accuracy": BinaryAccuracy(threshold=threshold),
        "precision": BinaryPrecision(threshold=threshold),
        "recall": BinaryRecall(threshold=threshold),
        "f1": BinaryF1Score(threshold=threshold),
        "auroc": BinaryAUROC()
    }

    results = {
        name: metric(y_prob, y_true).item()
        for name, metric in metrics.items()
    }

    return results


def run_metrics_pd(df: pd.DataFrame,
                   threshold: float = 0.5,
                   save_results: bool = True,
                   save_directory: str | Path | None = None):
    if save_results:
        if save_directory is None:
            raise ValueError("save_directory not specified")
        else:
            save_directory: Path = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
    y_prob = torch.tensor(df[ColumnNames.predictions].values, dtype=torch.float)
    y_true = torch.tensor(df[ColumnNames.labels].values, dtype=torch.int)

    results = run_metrics(y_prob, y_true, threshold=threshold)
    if save_results:
        with open(save_directory / "metrics.json", "w") as f:
            json.dump(results, f, indent=4)
    return results


def run_metrics_csv(csv_path: str | Path,
                    threshold: float = 0.5,
                    save_results: bool = True,
                    save_directory: str | Path = None):
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
    return run_metrics_pd(df, threshold=threshold, save_results=save_results, save_directory=save_directory)
