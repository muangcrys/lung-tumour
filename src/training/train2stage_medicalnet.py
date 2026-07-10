from training.training import *
from pretrains.medicalnet import get_pretrained_medicalnet, replace_medicalnet_classifier
from utility.reproducibility import reset_seed
from training.dataloader import get_train_val_loaders
from typing import Literal
import json


def train2stage_medicalnet(
        depth: Literal[18, 34, 50] = 18,
        ckt_path: str | Path | None = None,
        ckt_num_classes: int = None,
        replace_classifier: bool = True,
        train_annotation: str | Path | None = None,
        train_image_dir: str | Path | None = None,
        validate_annotation: str | Path | None = None,
        validate_image_dir: str | Path | None = None,
        first_stage_epochs: int = 20,
        second_stage_epochs: int = 50,
        lr1: float = 1e-4,
        lr2: float = 5e-5,
        decay1: float = 1e-4,
        decay2: float = 1e-4,
        bce_weight1: float = 2.0,
        bce_weight2: float = 2.0,
        metric: Literal[
            "loss", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"] = "auroc",
        higher_is_better: bool = None,
        threshold: float = 0.5,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 4242,
        deterministic: bool = True,
        report_frequency: int = 5,
        save_checkpoints: bool = True,
        base_directory: str | Path | None = None,
        device: torch.device | str | None = None,
        **kwargs
):
    if deterministic:
        reset_seed(seed)

    # load pretrained model
    model = get_pretrained_medicalnet(depth=depth, ckt_path=ckt_path)
    if replace_classifier:
        replace_medicalnet_classifier(model)

    # loaders
    train_loader, validate_loader = get_train_val_loaders(
        model_type="medical_pretrained",
        train_annotation=train_annotation,
        train_image_dir=train_image_dir,
        validate_annotation=validate_annotation,
        validate_image_dir=validate_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        deterministic=deterministic)

    # resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # resolve training save
    if save_checkpoints:
        save_directory = resolve_save_directory(model,
                                                base_directory=base_directory,
                                                model_string=f"medicalnet-{depth}-2stage",
                                                training="2stage",
                                                k=kwargs.get("k", -1),
                                                time_stamp=kwargs.get("time_stamp", None),)
        save_directory.mkdir(parents=True, exist_ok=True)
        # save training params
        training_configs_target = save_directory / FileNameResolver.get_training_configs_filename()
        training_configs = {
            "model": "Medicalnet",
            "training": "2stage",
            "depth": depth,
            "ckt_num_classes": ckt_num_classes,
            "replace_classifier": replace_classifier,
            "first_stage_epochs": first_stage_epochs,
            "second_stage_epochs": second_stage_epochs,
            "lr1": lr1,
            "lr2": lr2,
            "decay1": decay1,
            "decay2": decay2,
            "metric": metric,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "deterministic": deterministic,
            "device": str(device),
        }
        with open(training_configs_target, "w") as f:
            json.dump(training_configs, f, indent=4)

    else:
        save_directory = None

    # train loop
    best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_metrics, best_epoch, stats_dataframe = training_2_stage(
        model=model,
        train_loader=train_loader,
        validate_loader=validate_loader,
        model_type="medicalnet",
        lr1=lr1,
        lr2=lr2,
        decay1=decay1,
        decay2=decay2,
        bce_weight1=bce_weight1,
        bce_weight2=bce_weight2,
        metric=metric,
        higher_is_better=higher_is_better,
        threshold=threshold,
        frequency=report_frequency,
        save_checkpoints=save_checkpoints,
        save_directory=save_directory,
        device=device,
    )

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_metrics, best_epoch, stats_dataframe
