from training.training import *
from pretrains.resnet3d import *
from utility.reproducibility import reset_seed
from training.dataloader import get_train_val_loaders
import json

def train_resnet3d(
        depth: Literal[18, 34, 50] = 18,
        ckt_path: str|Path|None = None,
        ckt_num_classes: int = None,
        replace_classifier: bool = True,
        train_classifier_only: bool = False,
        train_annotation: str|Path|None = None,
        train_image_dir: str|Path|None = None,
        validate_annotation: str|Path|None = None,
        validate_image_dir: str|Path|None = None,
        epochs: int = 50,
        optim_type: Literal["AdamW"] = "AdamW",
        learning_rate: float = 1e-4,
        decay: float = 1e-5,
        pos_weight: float = 2.0,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 4242,
        deterministic: bool = True,
        report_frequency: int = 10,
        save_checkpoints: bool = True,
        base_directory: str|Path|None = None,
        device: torch.device | str | None = None,
        **kwargs
):
    if deterministic:
        reset_seed(seed)

    # load pretrained model
    model = get_pretrained_r3d(depth=depth, num_classes=ckt_num_classes, ckt_path=ckt_path)
    if replace_classifier:
        replace_resnet3d_classifier(model)

    if train_classifier_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # loss and optimizer
    criterion = get_BCE_loss(pos_weight=pos_weight)
    optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=decay, type=optim_type)


    # loaders
    train_loader, validate_loader = get_train_val_loaders(
        model_type="video_pretrained",
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
        save_directory = resolve_save_directory(model, base_directory=base_directory, model_string=f"resnet3d-{depth}")
        save_directory.mkdir(parents=True, exist_ok=True)
        # save training params
        training_configs_target = save_directory / FileNameResolver.get_training_configs_filename()
        training_configs = {
            "model": "Resnet3d",
            "depth": depth,
            "ckt_num_classes": ckt_num_classes,
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
            "device": str(device)
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
    )

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_metrics, best_epoch, stats_dataframe