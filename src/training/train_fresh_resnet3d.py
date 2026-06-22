from training.training import *
from models.fresh_resnet3d import *
from utility.reproducibility import reset_seed
from training.dataloader import get_train_val_loaders
import json

def train_fresh_resnet3d(
        depth: Literal[18, 34, 50] = 18,
        num_channels: int = None,
        ckt_path: str|Path|None = None,
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
        report_frequency: int = 10,
        save_checkpoints: bool = True,
        base_directory: str|Path|None = None,
        device: torch.device | str | None = None,
        **kwargs
):
    if deterministic:
        reset_seed(seed)

    if num_channels is None:
        num_channels = 3

    model = get_fresh_resnet3d(depth=depth, n_input_channels=num_channels)
    
    if ckt_path is not None:
        print(f"Loading weights from checkpoint: {ckt_path}")
        ckt = torch.load(ckt_path)
        state_dict = ckt['state_dict']
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint provided, training from scratch.")

    # loss and optimizer
    criterion = get_BCE_loss(pos_weight=pos_weight)
    optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=decay, type=optim_type)


    # loaders
    train_loader, validate_loader = get_train_val_loaders(
        model_type="random_init",
        n_input_channels=num_channels,
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
                                                model_string=f"fresh-resnet3d-{depth}-{num_channels}ch",
                                                k=kwargs.get("k", -1),
                                                time_stamp=kwargs.get("time_stamp", None), )
        save_directory.mkdir(parents=True, exist_ok=True)
        # save training params
        training_configs_target = save_directory / FileNameResolver.get_training_configs_filename()
        training_configs = {
            "model": "Fresh_ResNet3D" if ckt_path is None else f"ResNet3D(from checkpoint: {ckt_path})",
            "depth": depth,
            "epochs": epochs,
            "metric": metric,
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
        metric=metric
    )

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_metrics, best_epoch, stats_dataframe