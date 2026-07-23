from typing import Mapping, Any
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from evaluate.metrics import run_metrics_pd
from evaluate.precision_recall import plot_pr_curve_from_df
from evaluate.roc import plot_roc_from_df
from evaluate.confusion_matrix import plot_confusion_matrix
from evaluate.names import ColumnNames


def run_inference(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  model_type: str = None,
                  forward_args: Mapping[str, Any] = None,
                  save_predictions: bool = True,
                  save_directory: str | Path = None,
                  save_prefix: str = None,
                  file_name: str = None,
                  output_logits: bool = False,
                  device: str | torch.device = None):
    print(f"Running inference with model {model.__class__.__module__}:{model.__class__.__name__}")
    if save_predictions:
        print(f"Saving predictions to {save_directory}")
        if save_directory is None:
            raise ValueError("save_directory cannot be None")
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = "predictions.csv"
        if save_prefix is not None:
            file_name = f"{save_prefix}_{file_name}"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")

    if forward_args is None:
        forward_args = {}

    predictions = []
    labels = []

    # run inference
    model.eval()
    model.to(device)
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            labels.extend(y.cpu().reshape(-1).tolist())
            x = x.to(device)
            outputs = model(x, **forward_args)  # shape: (batch_size, 1)
            if model_type is not None and "vivit" in model_type.lower():
                logits = outputs.logits
            else:
                logits = outputs

            if output_logits:
                predictions.extend(logits.cpu().reshape(-1).tolist())
            else:
                predictions.extend(torch.sigmoid(logits).cpu().reshape(-1).tolist())

    # done
    df = pd.DataFrame({ColumnNames.labels: labels, ColumnNames.predictions: predictions})
    if save_predictions:
        print(f"Saving predictions to {save_directory}")
        df.to_csv(Path(save_directory) / file_name, index=False)

    return df


def run_inference_and_metrics(model: torch.nn.Module,
                              dataloader: torch.utils.data.DataLoader,
                              model_type: str = None,
                              forward_args: Mapping[str, Any] = None,
                              threshold: float = 0.5,
                              save_predictions: bool = True,
                              save_metrics: bool = True,
                              save_plots: bool = True,
                              prediction_file_name: str = None,
                              metrics_file_name: str = None,
                              roc_file_name: str = None,
                              pr_file_name: str = None,
                              conf_mat_file_name: str = None,
                              save_directory: str | Path = None,
                              save_prefix: str | None = None,
                              device: str | torch.device = None):
    # run inference
    df = run_inference(model=model,
                       data_loader=dataloader,
                       model_type=model_type,
                       forward_args=forward_args,
                       save_predictions=save_predictions,
                       save_directory=save_directory,
                       save_prefix=save_prefix,
                       file_name=prediction_file_name,
                       device=device,
                       output_logits=False)

    # run metrics
    metrics = run_metrics_pd(df=df,
                             threshold=threshold,
                             save_results=save_metrics,
                             save_directory=save_directory,
                             save_prefix=save_prefix,
                             file_name=metrics_file_name)

    # auroc plot, pr curve plot, conf mat
    auroc = metrics["auroc"]
    average_precision = metrics["average_precision"]
    conf_dict = metrics["confusion_matrix"]

    roc_fig, roc_ax = plot_roc_from_df(df=df,
                                       auroc=auroc,
                                       save_plot = save_plots,
                                       save_directory = save_directory,
                                       save_prefix = save_prefix,
                                       file_name = roc_file_name)
    pr_fig, pr_ax = plot_pr_curve_from_df(df=df,
                                          average_precision=average_precision,
                                          save_plot = save_plots,
                                          save_directory = save_directory,
                                          save_prefix=save_prefix,
                                          file_name = pr_file_name)
    conf_fig, conf_ax = plot_confusion_matrix(confusion_matrix_dict=conf_dict,
                                              save_plot = save_plots,
                                              save_directory = save_directory,
                                              save_prefix=save_prefix,
                                              file_name=conf_mat_file_name)

    return metrics, df, (roc_fig, roc_ax), (pr_fig, pr_ax), (conf_fig, conf_ax)


