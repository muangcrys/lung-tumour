from pathlib import Path

import torch
from torchmetrics.classification import BinaryROC, BinaryAUROC
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluate.names import ColumnNames


def plot_roc(y_prob: torch.Tensor,
             y_true: torch.Tensor,
             figsize: tuple[int, int] = (7,7),
             auroc: float | None = None,):

    # resolve auroc
    if auroc is None:
        auruc = BinaryAUROC()(y_prob, y_true).item()

    # roc
    fpr, tpr, thresholds = BinaryROC()(y_prob, y_true)
    roc_df = pd.DataFrame({
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    })

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=roc_df,
                 x='fpr',
                 y='tpr',
                 ax=ax,
                 label='ROC')
    sns.lineplot(x=[0, 1],
                 y=[0, 1],
                 linestyle='--',
                 ax=ax,
                 alpha=0.5,
                 label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (AUC = {auroc: .3f}')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig,ax


def plot_roc_from_df(df: pd.DataFrame,
                     predictions_column: str = ColumnNames.predictions,
                     labels_column: str = ColumnNames.labels,
                     auroc: float | None = None,
                     figsize: tuple[int, int] = (7, 7),
                     save_plot: bool = True,
                     save_directory: str | Path = None,
                     file_name: str = None):

    # resolve
    if save_plot:
        if save_directory is None:
            raise ValueError("save_directory not specified")
        else:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = "roc.pdf"

    y_prob = torch.tensor(df[predictions_column].values, dtype=torch.float)
    y_true = torch.tensor(df[labels_column].values, dtype=torch.int)

    fig, ax = plot_roc(y_true=y_true,
                       y_prob=y_prob,
                       figsize=figsize,
                       auroc=auroc)

    if save_plot:
        fig.savefig(save_directory / file_name, bbox_inches='tight')
    return fig, ax

