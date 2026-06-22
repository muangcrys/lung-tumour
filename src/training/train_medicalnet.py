from training.training import *
from pretrains.medicalnet import *
from utility.reproducibility import reset_seed
from training.dataloader import get_train_val_loaders
import json

def train_medicalnet(
        depth: Literal[18, 34, 50] = 18,
        ckt_path: str|Path|None = None,
        replace_classifier: bool = True,
        train_classifier_only: bool = False,
        train_annotation: str|Path|None = None,
        train_image_dir: str|Path|None = None,
        validate_annotation: str|Path|None = None,
        validate_image_dir: str|Path|None = None,
        metric: Literal["loss", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"] = "f1",
        epochs: int = 50,
        optim_type: Literal["AdamW"] = "AdamW",
        learning_rate: float = 1e-4,
        decay: float = 1e-5,
        pos_weight: float = 2.0,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 4242,
        deterministic: bool = True,
        report_frequency: int = 5,
        save_checkpoints: bool = True,
        base_directory: str|Path|None = None,
        device: torch.device | str | None = None,
        **kwargs
):
    if deterministic:
        reset_seed(seed)

    # load pretrained model
    model = get_pretrained_medicalnet(depth=depth, ckt_path=ckt_path)
    if replace_classifier:
        replace_medicalnet_classifier(model)

    if train_classifier_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.conv_seg.parameters():
            param.requires_grad = True

    # loss and optimizer
    criterion = get_BCE_loss(pos_weight=pos_weight)
    optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=decay, type=optim_type)


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
                                                model_string=f"medicalnet-{depth}",
                                                k=kwargs.get("k", -1),
                                                time_stamp=kwargs.get("time_stamp", None), )
        save_directory.mkdir(parents=True, exist_ok=True)
        # save training params
        training_configs_target = save_directory / FileNameResolver.get_training_configs_filename()
        training_configs = {
            "model": "Medicalnet",
            "depth": depth,
            "replace_classifier": replace_classifier,
            "train_classifier_only": train_classifier_only,
            "epochs": epochs,
            "optim_type": optim_type,
            "learning_rate": learning_rate,
            "decay": decay,
            "pos_weight": pos_weight,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "deterministic": deterministic,
            "device": str(device),
            "metric": metric
        }
        with open(training_configs_target, "w") as f:
            json.dump(training_configs, f, indent=4)

    else:
        save_directory = None

    # train loop
    best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_metrics, best_epoch, stats_dataframe = training_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        validate_loader=validate_loader,
        frequency=report_frequency,
        epochs=epochs,
        save_checkpoints=save_checkpoints,
        save_directory=save_directory,
        device=device,
        metric=metric
    )

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_metrics, best_epoch, stats_dataframe