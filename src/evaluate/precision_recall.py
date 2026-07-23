from pathlib import Path
import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAveragePrecision
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluate.names import ColumnNames
from typing import Tuple


def plot_pr_curve(y_prob: torch.Tensor,
                  y_true: torch.Tensor,
                  figsize: tuple[int, int] = (7, 7),
                  average_precision: float | None = None):
    # resolve average precision
    if average_precision is None:
        average_precision = BinaryAveragePrecision()(y_prob, y_true).item()

    # pr curve
    precision, recall, thresholds = BinaryPrecisionRecallCurve()(y_prob, y_true)
    pr_df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
    })

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=pr_df,
                 x='recall',
                 y='precision',
                 ax=ax,
                 estimator=None,
                 errorbar=None)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (AP = {average_precision: .3f})')
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax

def plot_pr_curve_from_df(df: pd.DataFrame,
                          predictions_column: str = ColumnNames.predictions,
                          labels_column: str = ColumnNames.labels,
                          average_precision: float | None = None,
                          figsize: tuple[int, int] = (7, 7),
                          save_plot: bool = True,
                          save_directory: Path | str = None,
                          file_name: str = None):
    # resolve
    if save_plot:
        if save_directory is None:
            raise ValueError("save_directory not specified")
        else:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = "precision_recall.pdf"

    y_prob = torch.tensor(df[predictions_column].values, dtype=torch.float)
    y_true = torch.tensor(df[labels_column].values, dtype=torch.int)

    fig, ax = plot_pr_curve(y_prob, y_true, figsize=figsize, average_precision=average_precision)

    if save_plot:
        fig.savefig(save_directory / file_name, bbox_inches='tight')

    return fig, ax