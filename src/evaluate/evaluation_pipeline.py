from pathlib import Path
from typing import Literal

from matplotlib import pyplot as plt
import torch

from evaluate.names import MetricFiles
from plotting.metrics import plot_all_metrics_from_csv
from training.names import FileNameResolver
import json
from plotting.loss import plot_loss_curves_from_csv
from models.fresh_resnet3d import get_fresh_resnet3d
from models.fresh_medicalnet import get_fresh_medicalnet
from models.vivit import get_config, get_vivit
from evaluate.inference import run_inference_and_metrics
from training.dataloader import get_validate_loader


def run_evaluation_on_model_directory(
        model_directory: str | Path,
        annotation: str | Path | None = None,
        image_dir: str | Path | None = None,
        preprocessing: str = None,
        metrics_directory: str | Path = None,
        final_model: bool = True,
        best_model: bool = True,
        plot_loss: bool = True,
        plot_metrics: bool = True,
        threshold: float = 0.5,
        model_type: Literal["vivit", "medicalnet", "resnet3d"] = None,
        depth: int = None,
        channels: int = None,
        batch_size: int = 8,
        num_workers: int = 0,
        device: str | torch.device = None,
        **kwargs,
):
    # sentinel
    if not final_model and not best_model and not plot_loss:
        # doing nothing?
        print("No tasks, returning")
        return
    model_directory: Path = Path(model_directory)

    if metrics_directory is None:
        metrics_directory: Path = model_directory
    else:
        metrics_directory: Path = Path(metrics_directory)
        metrics_directory.mkdir(parents=True, exist_ok=True)

    # plot loss first
    if plot_loss:
        _ , best_epoch = find_best_model_ckt(model_directory=model_directory)
        loss_fig, loss_ax = plot_loss_on_model_directory(model_directory=model_directory,
                                                         save_directory=metrics_directory, 
                                                         best_epoch=best_epoch)
        plt.close(loss_fig)

    # plot metrics
    if plot_metrics:
        metrics_fig, metrics_ax = plot_metrics_on_model_directory(model_directory=model_directory,
                                                                  save_directory=metrics_directory,
                                                                  best_epoch=best_epoch)
        plt.close(metrics_fig)

    # resolve model
    # get training configs
    training_configs = get_training_configs_from_directory(model_directory)
    # resolve model type
    if model_type is None:
        model_type = resolve_model_type(training_configs)
    if model_type == "resnet3d" or model_type == "medicalnet":
        if depth is None:
            depth = resolve_depth(training_configs)
    if channels is None:
        channels = resolve_channels(training_configs)
    if preprocessing is None:
        preprocessing = resolve_preprocessing_methods(training_configs)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if best_model:
        best_m = get_model_for_evaluation(model_type=model_type, depth=depth, channels=channels)
        best_ckt_path, best_epoch = find_best_model_ckt(model_directory=model_directory)
        load_model_from_ckt(best_m, best_ckt_path)
        b_dataloader = get_validate_loader(model_type=preprocessing,
                                           n_input_channels=channels,
                                           annotation=annotation,
                                           image_dir=image_dir,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
        _ = run_inference_and_metrics(
            model=best_m,
            dataloader=b_dataloader,
            model_type=model_type,
            threshold=threshold,
            prediction_file_name=MetricFiles.get_best_prediction_filename(best_epoch),
            metrics_file_name=MetricFiles.get_best_metrics_filename(best_epoch),
            roc_file_name=MetricFiles.get_best_roc_filename(best_epoch),
            pr_file_name=MetricFiles.get_best_pr_filename(best_epoch),
            conf_mat_file_name=MetricFiles.get_best_confmat_filename(best_epoch),
            save_directory=metrics_directory,
            device=device,
        )

    if final_model:
        final_m = get_model_for_evaluation(model_type=model_type, depth=depth, channels=channels)
        final_ckt_path, final_epoch = find_final_model_ckt(model_directory=model_directory)
        load_model_from_ckt(final_m, final_ckt_path)
        f_dataloader = get_validate_loader(model_type=preprocessing,
                                           n_input_channels=channels,
                                           annotation=annotation,
                                           image_dir=image_dir,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
        _ = run_inference_and_metrics(
            model=final_m,
            dataloader=f_dataloader,
            model_type=model_type,
            threshold=threshold,
            prediction_file_name=MetricFiles.get_final_prediction_filename(final_epoch),
            metrics_file_name=MetricFiles.get_final_metrics_filename(final_epoch),
            roc_file_name=MetricFiles.get_final_roc_filename(final_epoch),
            pr_file_name=MetricFiles.get_final_pr_filename(final_epoch),
            conf_mat_file_name=MetricFiles.get_final_confmat_filename(final_epoch),
            save_directory=metrics_directory,
            device=device,
        )

def get_training_configs_from_directory(
        model_directory: str | Path,
):
    model_directory: Path = Path(model_directory)
    config_file = model_directory / FileNameResolver.get_training_configs_filename()
    if not config_file.is_file():
        raise FileNotFoundError(f"Could not find config file at {config_file}")

    with open(config_file, "r") as f:
        training_configs = json.load(f)

    return training_configs


def plot_loss_on_model_directory(
        model_directory: Path,
        save_directory: str | Path = None,
        figsize: tuple[int, int] = (7, 7),
        best_epoch: int = None,
):
    model_directory: Path = Path(model_directory)
    if save_directory is None:
        save_directory: Path = model_directory
    else:
        save_directory: Path = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

    # find location of final loss file
    training_configs = get_training_configs_from_directory(model_directory)
    final_epoch = training_configs["epochs"]

    loss_file_name = FileNameResolver.get_final_stats_filename(final_epoch)
    loss_file = model_directory / loss_file_name

    # plot loss from csv
    fig, ax = plot_loss_curves_from_csv(csv_path=loss_file,
                                        save_plot=True,
                                        save_directory=model_directory,
                                        file_name=MetricFiles.get_final_loss_curve_filename(final_epoch),
                                        figsize=figsize,
                                        best_epoch=best_epoch)
    return fig, ax

def plot_metrics_on_model_directory(
        model_directory: Path,
        save_directory: str | Path = None,
        figsize: tuple[int, int] = (7, 7),
        best_epoch: int = None,
):
    model_directory: Path = Path(model_directory)
    if save_directory is None:
        save_directory: Path = model_directory
    else:
        save_directory: Path = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

    # find location of final loss file
    training_configs = get_training_configs_from_directory(model_directory)
    final_epoch = training_configs["epochs"]

    metrics_file_name = FileNameResolver.get_final_stats_filename(final_epoch)
    metrics_file = model_directory / metrics_file_name

    # plot
    fig, ax = plot_all_metrics_from_csv(csv_path=metrics_file,
                                        figsize=figsize,
                                        save_plot=True,
                                        save_directory=save_directory,
                                        file_name=MetricFiles.get_final_metrics_plot_filename(final_epoch),
                                        best_epoch=best_epoch)
    return fig, ax


def resolve_model_type(training_configs: dict) -> str:
    model_type: str = training_configs["model_type"].lower()
    if "resnet3d" in model_type or "resnet-3d" in model_type:
        return "resnet3d"
    elif "medicalnet" in model_type:
        return "medicalnet"
    elif "vivit" in model_type:
        return "vivit"
    else:
        raise ValueError(f"Could not resolve model type from training configs: {training_configs}")


def resolve_depth(training_configs: dict) -> int:
    return training_configs["depth"]


def resolve_channels(training_configs: dict) -> int:
    model_type: str = training_configs["model_type"].lower()
    if "resnet3d" in model_type:
        return 3
    elif "vivit_pretrained" == model_type:
        return 3
    elif "3ch" in model_type:
        return 3
    else:
        return 1


def resolve_preprocessing_methods(training_configs: dict,) -> str:
    model_string = training_configs["model_type"].lower()
    if "fresh" in model_string:
        return "random_init"
    elif model_string == "resnet3d":
        return "video_pretrained"
    elif "vivit_pretrained" == model_string:
        return "vivit_pretrained"
    elif "vivit_random" == model_string:
        return "vivit_random"
    elif "vivit_random_3ch" == model_string:
        return "vivit_random_3ch"
    # otherwise
    return "medical_pretrained"


def get_model_for_evaluation(
        model_type: str,
        depth: Literal[18, 34, 50],
        channels: int,
):
    if model_type == "resnet3d":
        model = get_fresh_resnet3d(depth, channels)
    elif model_type == "medicalnet":
        model = get_fresh_medicalnet(depth)
    elif model_type == "vivit":
        config = get_config(num_channels=channels)
        model = get_vivit(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def find_best_model_ckt(model_directory: Path | str, ):
    model_directory: Path = Path(model_directory)
    best_file_name = FileNameResolver.get_best_checkpoint_name("*")
    best_ckt_files = list(model_directory.glob(best_file_name))
    if len(best_ckt_files) == 0:
        raise FileNotFoundError(f"Could not find best ckt file at {model_directory}/{best_file_name}")
    elif len(best_ckt_files) > 1:
        print(f"Found more than one best ckt file at {model_directory}/{best_file_name}")
        print(best_ckt_files)
        print(f"Picking the last file: {best_ckt_files[-1]}")
        best_ckt_file = best_ckt_files[-1]
    else:
        best_ckt_file = best_ckt_files[-1]

    best_epoch = get_epoch_number_from_file_name(best_ckt_file.stem)
    return best_ckt_file, best_epoch


def find_final_model_ckt(model_directory: Path | str, ):
    model_directory: Path = Path(model_directory)
    final_file_name = FileNameResolver.get_final_checkpoint_name("*")
    final_ckt_files = list(model_directory.glob(final_file_name))
    if len(final_ckt_files) == 0:
        raise FileNotFoundError(f"Could not find final ckt file at {model_directory}/{final_file_name}")
    elif len(final_ckt_files) > 1:
        print(f"Found more than one final ckt file at {model_directory}/{final_file_name}")
        print(final_ckt_files)
        print(f"Picking the last file: {final_ckt_files[-1]}")
        final_ckt_file = final_ckt_files[-1]
    else:
        final_ckt_file = final_ckt_files[-1]

    final_epoch = get_epoch_number_from_file_name(final_ckt_file.stem)
    return final_ckt_file, final_epoch


def get_epoch_number_from_file_name(file_name: str):
    return int(file_name.split("_")[1])


def load_model_from_ckt(model: torch.nn.Module,
                        ckt_path: Path, ):
    ckt = torch.load(ckt_path)
    model.load_state_dict(ckt["model_state_dict"], strict=True)


