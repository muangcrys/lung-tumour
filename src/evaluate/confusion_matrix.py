from pathlib import Path

from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryConfusionMatrix
import torch
from typing import Literal
import seaborn as sns
import numpy as np


def run_confusion_matrix(y_prob: torch.Tensor,
                         y_true: torch.Tensor,
                         threshold: float = 0.5,
                         return_type: Literal['dict', 'tensor'] = 'dict'):
    confusion_matrix = BinaryConfusionMatrix(threshold=threshold)(y_prob, y_true)
    if return_type == 'tensor':
        return confusion_matrix
    elif return_type == 'dict':
        tn = confusion_matrix[0, 0].item()
        fp = confusion_matrix[0, 1].item()
        fn = confusion_matrix[1, 0].item()
        tp = confusion_matrix[1, 1].item()
        conf_mat_dict = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
        return conf_mat_dict
    else:
        raise ValueError(f"Unrecognized return_type {return_type}")


def reconstruct_confusion_matrix_tensor(confusion_matrix_dict: dict, ):
    tn = confusion_matrix_dict['TN']
    fp = confusion_matrix_dict['FP']
    fn = confusion_matrix_dict['FN']
    tp = confusion_matrix_dict['TP']

    return torch.tensor([[tn, fp],
                         [fn, tp]], dtype=torch.int)


def plot_confusion_matrix(confusion_matrix_tensor: torch.Tensor = None,
                          confusion_matrix_dict: dict = None,
                          figsize: tuple[int, int] = (3,3),
                          save_plot: bool = True,
                          save_directory: str | Path = None,
                          file_name: str = None,):
    if confusion_matrix_dict is None and confusion_matrix_tensor is None:
        raise ValueError("Must supply at least a tensor or a dict that represents a confusion matrix")
    if confusion_matrix_tensor is None:
        confusion_matrix_tensor = reconstruct_confusion_matrix_tensor(confusion_matrix_dict)

    if save_plot:
        if save_directory is None:
            raise ValueError("save_directory not specified")
        if file_name is None:
            file_name = "confusion_matrix.pdf"
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)


    # np
    cm = confusion_matrix_tensor.numpy()
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    
    annot = np.array([
        [f"{pct:.1%}\n({count})" for count, pct in zip(row_count, row_pct)]
        for row_count, row_pct in zip(cm, cm_pct)
    ])

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_pct,
            annot=annot,
            fmt="",
            cmap="magma",
            cbar=False,
            annot_kws={"size": 10},
            xticklabels=["-", "+"],
            yticklabels=["-", "+"],
            ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if save_plot:
        fig.savefig(save_directory / file_name, bbox_inches="tight")
    return fig, ax


