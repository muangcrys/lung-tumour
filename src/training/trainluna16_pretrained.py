from transformers import VivitConfig

from training.training import *
from pretrains.resnet3d import *
from pretrains.medicalnet import *
import pretrains.vivit
from utility.reproducibility import reset_seed
from training.dataloader import get_train_val_loaders
from typing import Literal
import json


def train_luna16_pretrained(
        model_type: Literal["resnet3d", "medicalnet", "vivit"],
        model_config: VivitConfig = None,
        depth: Literal[18, 34, 50] = 18,
        ckt_path: str | Path | None = None,
        ckt_num_classes: int = None,
        replace_classifier: bool = True,
        luna16_train_annotation: str | Path | None = None,
        luna16_train_image_dir: str | Path | None = None,
        luna16_validate_annotation: str | Path | None = None,
        luna16_validate_image_dir: str | Path | None = None,
        luna25_train_annotation: str | Path | None = None,
        luna25_train_image_dir: str | Path | None = None,
        luna25_validate_annotation: str | Path | None = None,
        luna25_validate_image_dir: str | Path | None = None,
        first_stage_epochs: int = 30,
        second_stage_epochs: int = 30,
        lr1: float = 5e-5,
        lr2: float = 5e-5,
        decay1: float = 1e-3,
        decay2: float = 1e-3,
        luna25_weight: float = 2.0,
        metric: Literal[
            "loss", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"] = "auroc",
        higher_is_better: bool = None,
        threshold: float = 0.5,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 4242,
        deterministic: bool = True,
        report_frequency: int = 10,
        save_checkpoints: bool = True,
        base_directory: str | Path | None = None,
        device: torch.device | str | None = None,
        preprocessing_pipeline: Literal["video_pretrained", "medical_pretrained", "random_init", "vivit_pretrained", "vivit_random"] = "random_init",
        n_input_channels: int = 1,
        **kwargs
):
    if deterministic:
        reset_seed(seed)

    # load pretrained model
    model_type = model_type.lower()
    if model_type == "resnet3d":
        model = get_pretrained_r3d(depth=depth, num_classes=ckt_num_classes, ckt_path=ckt_path)
        if replace_classifier:
            replace_resnet3d_classifier(model)
    elif model_type == "medicalnet":
        model = get_pretrained_medicalnet(depth=depth, ckt_path=ckt_path)
        if replace_classifier:
            replace_medicalnet_classifier(model)
    elif model_type == "vivit":
        model = pretrains.vivit.get_model_pretrained(model_config)

    # loaders
    luna16_train_loader, luna16_validate_loader = get_train_val_loaders(
        model_type=preprocessing_pipeline,
        train_annotation=luna16_train_annotation,
        train_image_dir=luna16_train_image_dir,
        validate_annotation=luna16_validate_annotation,
        validate_image_dir=luna16_validate_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        deterministic=deterministic,
        use_sampler=False,
        n_input_channels=n_input_channels
    )
    luna25_train_loader, luna25_validate_loader = get_train_val_loaders(
        model_type=preprocessing_pipeline,
        train_annotation=luna25_train_annotation,
        train_image_dir=luna25_train_image_dir,
        validate_annotation=luna25_validate_annotation,
        validate_image_dir=luna25_validate_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        deterministic=deterministic,
        pos_weight=2.0,
        use_sampler=True,
        n_input_channels=n_input_channels
    )

    # resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # resolve training save
    if save_checkpoints:
        if model_type == "resnet3d":
            model_string = f"resnet3d-{depth}-luna16double"
        elif model_type == "medicalnet":
            model_string = f"medicalnet-{depth}-luna16double"
        elif model_type == "vivit":
            model_string = f"vivit_pretrained-luna16double"
        else:
            raise NotImplementedError()
        save_directory = resolve_save_directory(model,
                                                base_directory=base_directory,
                                                model_string=model_string,
                                                training="luna16_double",
                                                k=kwargs.get("k", -1),
                                                time_stamp=kwargs.get("time_stamp", None), )
        save_directory.mkdir(parents=True, exist_ok=True)
        # save training params
        training_configs_target = save_directory / FileNameResolver.get_training_configs_filename()
        if model_type == "resnet3d":
            model_config = f"ResNet3d"
        elif model_type == "medicalnet":
            model_config = f"Medicalnet"
        elif model_type == "vivit":
            model_config = f"vivit_pretrained"
        else:
            raise NotImplementedError()
        training_configs = {
            "model": model_config,
            "training": "luna16_double",
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
    best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_metrics, best_epoch, stats_dataframe = training_luna16_double(
        model=model,
        luna16_train_loader=luna16_train_loader,
        luna16_validate_loader=luna16_validate_loader,
        luna25_train_loader=luna25_train_loader,
        luna25_validate_loader=luna25_validate_loader,
        model_type=model_type,
        lr1=lr1,
        lr2=lr2,
        decay1=decay1,
        decay2=decay2,
        first_stage_epochs=first_stage_epochs,
        second_stage_epochs=second_stage_epochs,
        luna25_weight=luna25_weight,
        metric=metric,
        higher_is_better=higher_is_better,
        threshold=threshold,
        frequency=report_frequency,
        save_checkpoints=save_checkpoints,
        save_directory=save_directory,
        device=device,
    )

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_metrics, best_epoch, stats_dataframe
