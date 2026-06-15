from training.training import *
from pretrains.resnet3d import *
from utility.reproducibility import reset_seed
from training.dataloader import get_train_val_loaders
import models.vivit
import pretrains.vivit
from transformers import VivitConfig


def train_vivit(
        model_config: VivitConfig = None,
        ckt_path: str|Path|None = None,
        pretrained: bool = True,
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
        **kwargs,
):
    if deterministic:
        reset_seed()

    # load model
    if pretrained:
        print("Loading pretrained model")
        model = pretrains.vivit.get_model_pretrained(model_config)
    else:
        print("Loading fresh vivit model")
        model = models.vivit.get_vivit(model_config)

    # load checkpoint if specified
    if ckt_path is None:
        print("No checkpoint provided -> training from scratch (or from pretrained).")
    else:
        # checkpoint loading logic here
        ckt_path = Path(ckt_path)
        print(f"Loading checkpoint from {ckt_path}")
        ckt = torch.load(ckt_path)
        state_dict = ckt['state_dict']
        model.load_state_dict(state_dict)

    if train_classifier_only:
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        # unfreeze position embeddings
        # model.vivit.embeddings.cls_token.requires_grad = True
        model.vivit.embeddings.position_embeddings.requires_grad = True

    # loss and optimizer
    criterion = get_BCE_loss(pos_weight=pos_weight)
    optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=decay, type=optim_type)

    # dataloaders
    model_type = "vivit_pretrained" if pretrained else "vivit_random"
    train_loader, validate_loader = get_train_val_loaders(
        model_type=model_type,
        train_annotation=train_annotation,
        train_image_dir=train_image_dir,
        validate_annotation=validate_annotation,
        validate_image_dir=validate_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        deterministic=deterministic,
    )

    # device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # resolve training save
    if save_checkpoints:
        save_directory = resolve_save_directory(model, base_directory, model_string=model_type)
        save_directory.mkdir(parents=True, exist_ok=True)
        # save training params
        training_configs_target = save_directory / FileNameResolver.get_training_configs_filename()
        training_configs = {
            "model": model_type,
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
            "model_config": model.config.to_dict()
        }
    else:
        save_directory = None

    # train_loop
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
        vivit=True
    )

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_metrics, best_epoch, stats_dataframe