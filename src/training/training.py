import torch
from torch import nn
from torch.optim import AdamW
from typing import Literal, Mapping, Any
from pathlib import Path
from datetime import datetime
from evaluate.metrics import run_metrics
from training.names import FileNameResolver, ClassifierOnlyFileNameResolver, LUNA16FileNameResolver
from utility.paths import PathList
from utility.utils import get_timestamp_now
from tqdm.auto import tqdm
from copy import deepcopy
import pandas as pd
import json


def get_vivit_forward_args():
    return {
        "interpolate_pos_encoding": True
    }


def get_optimizer(model: torch.nn.Module,
                  type: Literal["AdamW"] = "AdamW",
                  learning_rate: float = 1e-4,
                  weight_decay: float = 1e-5, ):
    if type == "AdamW":
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {type}")

    return optimizer


def get_BCE_loss(pos_weight: float = 2.0):
    weight = torch.tensor([pos_weight])
    return nn.BCEWithLogitsLoss(weight=weight)


def resolve_save_directory(model: torch.nn.Module,
                           base_directory: str | Path = None,
                           model_string: str = None,
                           training: Literal["normal", "2stage", "luna16_double"] = "normal",
                           time_stamp: str = None,
                           k: int = -1):
    if time_stamp is None:
        # time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        time_stamp = get_timestamp_now()
    if model_string is None:
        model_string = f"{model.__class__.__module__}-{model.__class__.__name__}"

    is_k_fold: bool = k is not None and k > 0

    if base_directory is None:
        if training == "normal":
            base_directory = PathList.saved_weights_dir.resolve() if not is_k_fold else PathList.saved_kfold_weights_dir.resolve()
        elif training == "2stage":
            base_directory = PathList.saved_2stage_weights_dir.resolve() if not is_k_fold else PathList.saved_kfold_2stage_weights_dir.resolve()
        elif training == "luna16_double":
            base_directory = PathList.saved_luna16_weights_dir.resolve() if not is_k_fold else PathList.saved_kfold_luna16_weights_dir.resolve()
        else:
            raise NotImplementedError(f"Unknown training type: {training}")
    else:
        base_directory = Path(base_directory).resolve()
    if is_k_fold:
        return base_directory / model_string / time_stamp / f"fold_{k}"
    else:
        return base_directory / model_string / time_stamp


def training_loop(model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  criterion: torch.nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  validate_loader: torch.utils.data.DataLoader,
                  metric: Literal["loss", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"] = "f1",
                  higher_is_better: bool = None,
                  threshold: float = 0.5,
                  forward_args: Mapping[str, Any] = None,
                  frequency: int = 10,
                  epochs: int = 100,
                  save_checkpoints: bool = True,
                  save_directory: str | Path = None,
                  device: torch.device | str = None,
                  vivit: bool = False, ):
    print(f"Model Name: {model.__class__.__module__} -> {model.__class__.__name__}")
    print(f"# Epochs: {epochs}")
    print("Initializing training loop...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Training will be run on {device}")

    if save_checkpoints:
        if save_directory is None:
            save_directory: Path = resolve_save_directory(model, save_directory)
        else:
            save_directory: Path = Path(save_directory).resolve()
        print(f"Saving weights and statistics to {save_directory} every {frequency} epochs")
        if not save_directory.is_dir():
            save_directory.mkdir(parents=True, exist_ok=True)

    if forward_args is None:
        forward_args = dict()

    # start
    model.to(device)
    criterion.to(device)

    best_model_state_dict = None
    best_optimizer_state_dict = None
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_metrics = None

    # resolve merics
    if higher_is_better is None:
        higher_is_better = metric != "loss"

    if higher_is_better:
        best_metric = float("-inf")
    else:
        # lower is better
        best_metric = float("inf")
    best_epoch = -1

    # dataframe to track training loss, validation loss, metrics
    stats_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "val_loss",
                                            "accuracy", "precision", "recall", "f1", "auroc", "average_precision"])

    # set average training and validation loss for ref
    avg_train_loss = None
    avg_val_loss = None
    epoch_metrics = None

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            y_pred = model(inputs, **forward_args)
            if vivit:
                y_pred = y_pred.logits

            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        labels_tensor = torch.empty(0, dtype=torch.int).cpu()
        predictions_tensor = torch.empty(0).cpu()

        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                y_pred = model(inputs, **forward_args)
                if vivit:
                    y_pred = y_pred.logits
                loss = criterion(y_pred, labels)
                val_loss += loss.item()

                # append
                y_prob = torch.sigmoid(y_pred).cpu().reshape(-1)
                labels_tensor = torch.cat((labels_tensor, labels.cpu().int().reshape(-1)), 0)
                predictions_tensor = torch.cat((predictions_tensor, y_prob), 0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(validate_loader)

        epoch_metrics = run_metrics(y_prob=predictions_tensor,
                                    y_true=labels_tensor,
                                    threshold=threshold,
                                    skip_confusion_matrix=True)

        # report
        print("=" * 20)
        print(f"EPOCH: {epoch + 1}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Accuracy: {epoch_metrics['accuracy']}")
        print(f"Precision: {epoch_metrics['precision']}")
        print(f"Recall: {epoch_metrics['recall']}")
        print(f"F1 Score: {epoch_metrics['f1']}")
        print(f"AUROC: {epoch_metrics['auroc']}")
        print(f"Average Precision: {epoch_metrics['average_precision']}")

        # update pd
        stats_dataframe.loc[len(stats_dataframe)] = [epoch + 1, avg_train_loss, avg_val_loss,
                                                     epoch_metrics['accuracy'],
                                                     epoch_metrics['precision'],
                                                     epoch_metrics['recall'],
                                                     epoch_metrics['f1'],
                                                     epoch_metrics['auroc'],
                                                     epoch_metrics['average_precision']]

        if metric == "loss":
            this_epoch_metric = avg_val_loss
        else:
            this_epoch_metric = epoch_metrics[metric]

        if higher_is_better:
            improvement = this_epoch_metric > best_metric
        else:
            # lower is better
            improvement = this_epoch_metric < best_metric
        if improvement:
            print(
                f"Model improved on validation metric: {metric}, saving best model state dict and optimizer state dict")
            best_val_loss = avg_val_loss
            best_model_state_dict = deepcopy(model.state_dict())
            best_optimizer_state_dict = deepcopy(optimizer.state_dict())
            best_train_loss = avg_train_loss
            best_epoch = epoch
            best_metric = this_epoch_metric
            best_metrics = epoch_metrics

        if save_checkpoints and (epoch + 1) % frequency == 0:
            checkpoint_target = save_directory / FileNameResolver.get_checkpoint_name(epoch + 1)
            print(f"Saving checkpoint to {checkpoint_target}")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, checkpoint_target)
            statistics_target = save_directory / FileNameResolver.get_stats_filename(epoch + 1)
            print(f"Saving statistics to {statistics_target}")
            stats_dataframe.to_csv(statistics_target, index=False)

    # after loop
    print("=" * 40)

    # save best model
    if best_model_state_dict is None or best_optimizer_state_dict is None:
        print("No checkpoint found for best model!")
    else:
        best_checkpoint_target = save_directory / FileNameResolver.get_best_checkpoint_name(best_epoch + 1)
        torch.save({
            "epoch": best_epoch + 1,
            "model_state_dict": best_model_state_dict,
            "optimizer_state_dict": best_optimizer_state_dict,
            "train_loss": best_train_loss,
            "val_loss": best_val_loss,
        }, best_checkpoint_target)

        best_json_target = save_directory / FileNameResolver.get_best_json_filename(best_epoch + 1)
        with open(best_json_target, "w") as f:
            best_json = {
                "epoch": best_epoch + 1,
                "train_loss": best_train_loss,
                "val_loss": best_val_loss,
                "metrics": best_metrics
            }
            json.dump(best_json, f, indent=4)

    # save last checkpoint
    final_checkpoint_target = save_directory / FileNameResolver.get_final_checkpoint_name(epochs)
    final_statistics_target = save_directory / FileNameResolver.get_final_stats_filename(epochs)
    if save_checkpoints and epochs % frequency == 0:
        # already saved in loop -> rename instead
        epoch_checkpoint_target = save_directory / FileNameResolver.get_checkpoint_name(epochs)
        epoch_statistics_target = save_directory / FileNameResolver.get_stats_filename(epochs)
        epoch_checkpoint_target.rename(final_checkpoint_target)
        epoch_statistics_target.rename(final_statistics_target)
    else:
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, final_checkpoint_target)
        stats_dataframe.to_csv(final_statistics_target, index=False)
    # json
    final_json_target = save_directory / FileNameResolver.get_final_json_filename(epochs)
    with open(final_json_target, "w") as f:
        final_json = {
            "epoch": epochs,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "metrics": epoch_metrics
        }
        json.dump(final_json, f, indent=4)

    # done
    print("=" * 40)
    print("Done!")

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_metrics, best_epoch, stats_dataframe


def training_2_stage(model: torch.nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     validate_loader: torch.utils.data.DataLoader,
                     model_type: Literal["resnet3d", "medicalnet", "vivit"] = "resnet3d",
                     lr1: float = 1e-3,
                     lr2: float = 5e-5,
                     decay1: float = 1e-4,
                     decay2: float = 1e-4,
                     bce_weight1: float = 2.0,
                     bce_weight2: float = 2.0,
                     metric: Literal[
                         "loss", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"] = "f1",
                     higher_is_better: bool = None,
                     threshold: float = 0.5,
                     forward_args: Mapping[str, Any] = None,
                     frequency: int = 10,
                     first_stage_epochs: int = 20,
                     second_stage_epochs: int = 50,
                     save_checkpoints: bool = True,
                     save_directory: str | Path = None,
                     device: torch.device | str = None, ):
    print(f"Model Name: {model.__class__.__module__} -> {model.__class__.__name__}")
    print(f"# Training classifier (and positional embeddings for ViViT) only for {first_stage_epochs} epochs")
    print(f"# Training entire model for {second_stage_epochs} epochs")
    print("Initializing training loop...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Training will be run on {device}")

    if save_checkpoints:
        if save_directory is None:
            save_directory: Path = resolve_save_directory(model, save_directory)
        else:
            save_directory: Path = Path(save_directory).resolve()
        print(f"Saving weights and statistics to {save_directory} every {frequency} epochs")
        if not save_directory.is_dir():
            save_directory.mkdir(parents=True, exist_ok=True)

    if forward_args is None:
        forward_args = dict()

    # model_type = model_type.lower()
    vivit = "vivit" in model_type

    print()
    print(("#" * 20) + "FIRST STAGE: TRAINING CLASSIFIER ONLY" + ("#" * 20))
    print()

    print("Initializing first stage training...")
    first_criterion = get_BCE_loss(pos_weight=bce_weight1).to(device)

    print("Freezing all model parameters except classifier (and positional embeddings if ViViT)...")
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze classifier
    # initializing criterion
    if model_type == "resnet3d":
        for param in model.fc.parameters():
            param.requires_grad = True
        first_stage_optimizer = AdamW(model.fc.parameters(), lr=lr1, weight_decay=decay1)
    elif model_type == "medicalnet":
        for param in model.conv_seg.parameters():
            param.requires_grad = True
        first_stage_optimizer = AdamW(model.conv_seg.parameters(), lr=lr1, weight_decay=decay1)
    elif model_type == "vivit":
        for param in model.classifier.parameters():
            # vivit classifier head
            param.requires_grad = True
        model.vivit.embeddings.position_embeddings.requires_grad = True
        first_stage_optimizer = AdamW([
            {"params": model.classifier.parameters(), "lr": lr1},
            {"params": [model.vivit.embeddings.position_embeddings], "lr": lr2},
        ], weight_decay=decay1)
    else:
        print(f"Unknown model type {model_type}, assuming it's resnet3d with classifier layer named fc")
        for param in model.fc.parameters():
            param.requires_grad = True
        first_stage_optimizer = AdamW(model.fc.parameters(), lr=lr1, weight_decay=decay1)

    first_best_model_state_dict = None
    first_best_optimizer_state_dict = None
    first_best_train_loss = float("inf")
    first_best_val_loss = float("inf")
    first_best_metrics = None

    # resolve merics
    if higher_is_better is None:
        higher_is_better = metric != "loss"

    if higher_is_better:
        first_best_metric = float("-inf")
    else:
        # lower is better
        first_best_metric = float("inf")
    first_best_epoch = -1

    # dataframe to track training loss, validation loss, metrics
    first_stats_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "val_loss",
                                                  "accuracy", "precision", "recall", "f1", "auroc",
                                                  "average_precision"])

    # set average training and validation loss for ref
    first_avg_train_loss = None
    first_avg_val_loss = None
    first_epoch_metrics = None

    model.to(device)

    for epoch in tqdm(range(first_stage_epochs), desc="First Stage Progress"):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            first_stage_optimizer.zero_grad()

            y_pred = model(inputs, **forward_args)
            if vivit:
                y_pred = y_pred.logits

            loss = first_criterion(y_pred, labels)
            loss.backward()
            first_stage_optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        labels_tensor = torch.empty(0, dtype=torch.int).cpu()
        predictions_tensor = torch.empty(0).cpu()

        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                y_pred = model(inputs, **forward_args)
                if vivit:
                    y_pred = y_pred.logits
                loss = first_criterion(y_pred, labels)
                val_loss += loss.item()

                # append
                y_prob = torch.sigmoid(y_pred).cpu().reshape(-1)
                labels_tensor = torch.cat((labels_tensor, labels.cpu().int().reshape(-1)), 0)
                predictions_tensor = torch.cat((predictions_tensor, y_prob), 0)

        first_avg_train_loss = train_loss / len(train_loader)
        first_avg_val_loss = val_loss / len(validate_loader)

        first_epoch_metrics = run_metrics(y_prob=predictions_tensor,
                                          y_true=labels_tensor,
                                          threshold=threshold,
                                          skip_confusion_matrix=True)

        # report
        print("=" * 20)
        print(f"EPOCH: {epoch + 1}")
        print(f"Train Loss: {first_avg_train_loss}")
        print(f"Validation Loss: {first_avg_val_loss}")
        print(f"Accuracy: {first_epoch_metrics['accuracy']}")
        print(f"Precision: {first_epoch_metrics['precision']}")
        print(f"Recall: {first_epoch_metrics['recall']}")
        print(f"F1 Score: {first_epoch_metrics['f1']}")
        print(f"AUROC: {first_epoch_metrics['auroc']}")
        print(f"Average Precision: {first_epoch_metrics['average_precision']}")

        # update pd
        first_stats_dataframe.loc[len(first_stats_dataframe)] = [epoch + 1,
                                                                 first_avg_train_loss,
                                                                 first_avg_val_loss,
                                                                 first_epoch_metrics['accuracy'],
                                                                 first_epoch_metrics['precision'],
                                                                 first_epoch_metrics['recall'],
                                                                 first_epoch_metrics['f1'],
                                                                 first_epoch_metrics['auroc'],
                                                                 first_epoch_metrics['average_precision']]

        if metric == "loss":
            this_epoch_metric = first_avg_val_loss
        else:
            this_epoch_metric = first_epoch_metrics[metric]

        if higher_is_better:
            improvement = this_epoch_metric > first_best_metric
        else:
            # lower is better
            improvement = this_epoch_metric < first_best_metric
        if improvement:
            print(
                f"Model improved on validation metric: {metric}, saving best model state dict and optimizer state dict")
            first_best_val_loss = first_avg_val_loss
            first_best_model_state_dict = deepcopy(model.state_dict())
            first_best_optimizer_state_dict = deepcopy(first_stage_optimizer.state_dict())
            first_best_train_loss = first_avg_train_loss
            first_best_epoch = epoch
            first_best_metric = this_epoch_metric
            first_best_metrics = first_epoch_metrics

        if save_checkpoints and (epoch + 1) % frequency == 0:
            checkpoint_target = save_directory / ClassifierOnlyFileNameResolver.get_checkpoint_name(epoch + 1)
            print(f"Saving checkpoint to {checkpoint_target}")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": first_stage_optimizer.state_dict(),
                "train_loss": first_avg_train_loss,
                "val_loss": first_avg_val_loss,
            }, checkpoint_target)
            statistics_target = save_directory / ClassifierOnlyFileNameResolver.get_stats_filename(epoch + 1)
            print(f"Saving statistics to {statistics_target}")
            first_stats_dataframe.to_csv(statistics_target, index=False)

    # after loop
    print("=" * 40)

    # save best model
    if first_best_model_state_dict is None or first_best_optimizer_state_dict is None:
        print("No checkpoint found for best model!")
        raise RuntimeError("No checkpoint found for best model!")
    else:
        best_checkpoint_target = save_directory / ClassifierOnlyFileNameResolver.get_best_checkpoint_name(
            first_best_epoch + 1)
        torch.save({
            "epoch": first_best_epoch + 1,
            "model_state_dict": first_best_model_state_dict,
            "optimizer_state_dict": first_best_optimizer_state_dict,
            "train_loss": first_best_train_loss,
            "val_loss": first_best_val_loss,
        }, best_checkpoint_target)

        best_json_target = save_directory / ClassifierOnlyFileNameResolver.get_best_json_filename(first_best_epoch + 1)
        with open(best_json_target, "w") as f:
            best_json = {
                "epoch": first_best_epoch + 1,
                "train_loss": first_best_train_loss,
                "val_loss": first_best_val_loss,
                "metrics": first_best_metrics
            }
            json.dump(best_json, f, indent=4)

    # save last checkpoint
    final_checkpoint_target = save_directory / ClassifierOnlyFileNameResolver.get_final_checkpoint_name(
        first_stage_epochs)
    final_statistics_target = save_directory / ClassifierOnlyFileNameResolver.get_final_stats_filename(
        first_stage_epochs)
    if save_checkpoints and first_stage_epochs % frequency == 0:
        # already saved in loop -> rename instead
        epoch_checkpoint_target = save_directory / ClassifierOnlyFileNameResolver.get_checkpoint_name(
            first_stage_epochs)
        epoch_statistics_target = save_directory / ClassifierOnlyFileNameResolver.get_stats_filename(first_stage_epochs)
        epoch_checkpoint_target.rename(final_checkpoint_target)
        epoch_statistics_target.rename(final_statistics_target)
    else:
        torch.save({
            "epoch": first_stage_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": first_stage_optimizer.state_dict(),
            "train_loss": first_avg_train_loss,
            "val_loss": first_avg_val_loss,
        }, final_checkpoint_target)
        first_stats_dataframe.to_csv(final_statistics_target, index=False)
    # json
    final_json_target = save_directory / ClassifierOnlyFileNameResolver.get_final_json_filename(first_stage_epochs)
    with open(final_json_target, "w") as f:
        final_json = {
            "epoch": first_stage_epochs,
            "train_loss": first_avg_train_loss,
            "val_loss": first_avg_val_loss,
            "metrics": first_epoch_metrics
        }
        json.dump(final_json, f, indent=4)

    # done
    print("First stage finished")
    print("=" * 40)
    print()
    print(("#" * 20) + "SECOND STAGE: ENTIRE MODEL FINETUNING" + ("#" * 20))
    print()
    # reporting
    print(f"Continuing from epoch: {first_best_epoch + 1}")
    print("Statistics:")
    print(f"Training Loss: {first_best_train_loss}")
    print(f"Validation Loss: {first_best_val_loss}")
    print(f"Metrics ({metric}): {first_best_metrics}")
    print("=" * 40)
    print("Initializing second stage finetuning...")

    print("Unfreezing model parameters...")
    # reverting model to best epoch
    model.load_state_dict(first_best_model_state_dict)

    # unfreeze all params
    for param in model.parameters():
        param.requires_grad = True

    # optim
    second_stage_optimizer = AdamW(model.parameters(),
                                   lr=lr2,
                                   weight_decay=decay2)

    # loss
    second_criterion = get_BCE_loss(pos_weight=bce_weight2).to(device)

    second_best_epoch = -1
    second_best_train_loss = first_best_train_loss
    second_best_val_loss = first_best_val_loss
    second_best_metric = first_best_metric
    second_best_metrics = first_best_metrics
    second_best_model_state_dict = deepcopy(first_best_model_state_dict)
    second_best_optimizer_state_dict = None

    # dataframe to track training loss, validation loss, metrics
    second_stats_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "val_loss",
                                                   "accuracy", "precision", "recall", "f1", "auroc",
                                                   "average_precision"])

    # set average training and validation loss for ref
    second_avg_train_loss = None
    second_avg_val_loss = None
    second_epoch_metrics = None

    model.to(device)

    for epoch in tqdm(range(second_stage_epochs), desc="Second Stage Progress"):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            second_stage_optimizer.zero_grad()

            y_pred = model(inputs, **forward_args)
            if vivit:
                y_pred = y_pred.logits

            loss = second_criterion(y_pred, labels)
            loss.backward()
            second_stage_optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        labels_tensor = torch.empty(0, dtype=torch.int).cpu()
        predictions_tensor = torch.empty(0).cpu()

        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                y_pred = model(inputs, **forward_args)
                if vivit:
                    y_pred = y_pred.logits
                loss = first_criterion(y_pred, labels)
                val_loss += loss.item()

                # append
                y_prob = torch.sigmoid(y_pred).cpu().reshape(-1)
                labels_tensor = torch.cat((labels_tensor, labels.cpu().int().reshape(-1)), 0)
                predictions_tensor = torch.cat((predictions_tensor, y_prob), 0)

        second_avg_train_loss = train_loss / len(train_loader)
        second_avg_val_loss = val_loss / len(validate_loader)

        second_epoch_metrics = run_metrics(y_prob=predictions_tensor,
                                           y_true=labels_tensor,
                                           threshold=threshold,
                                           skip_confusion_matrix=True)

        # report
        print("=" * 20)
        print(f"EPOCH: {epoch + 1}")
        print(f"Train Loss: {second_avg_train_loss}")
        print(f"Validation Loss: {second_avg_val_loss}")
        print(f"Accuracy: {second_epoch_metrics['accuracy']}")
        print(f"Precision: {second_epoch_metrics['precision']}")
        print(f"Recall: {second_epoch_metrics['recall']}")
        print(f"F1 Score: {second_epoch_metrics['f1']}")
        print(f"AUROC: {second_epoch_metrics['auroc']}")
        print(f"Average Precision: {second_epoch_metrics['average_precision']}")

        # update pd
        second_stats_dataframe.loc[len(second_stats_dataframe)] = [epoch + 1,
                                                                   second_avg_train_loss,
                                                                   second_avg_val_loss,
                                                                   second_epoch_metrics['accuracy'],
                                                                   second_epoch_metrics['precision'],
                                                                   second_epoch_metrics['recall'],
                                                                   second_epoch_metrics['f1'],
                                                                   second_epoch_metrics['auroc'],
                                                                   second_epoch_metrics['average_precision']]

        if metric == "loss":
            this_epoch_metric = second_avg_val_loss
        else:
            this_epoch_metric = second_epoch_metrics[metric]

        if higher_is_better:
            improvement = this_epoch_metric > second_best_metric
        else:
            # lower is better
            improvement = this_epoch_metric < second_best_metric
        if improvement:
            print(
                f"Model improved on validation metric: {metric}, saving best model state dict and optimizer state dict")
            second_best_val_loss = second_avg_val_loss
            second_best_model_state_dict = deepcopy(model.state_dict())
            second_best_optimizer_state_dict = deepcopy(second_stage_optimizer.state_dict())
            second_best_train_loss = second_avg_train_loss
            second_best_epoch = epoch
            second_best_metric = this_epoch_metric
            second_best_metrics = second_epoch_metrics

        if save_checkpoints and (epoch + 1) % frequency == 0:
            checkpoint_target = save_directory / FileNameResolver.get_checkpoint_name(epoch + 1)
            print(f"Saving checkpoint to {checkpoint_target}")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": second_stage_optimizer.state_dict(),
                "train_loss": second_avg_train_loss,
                "val_loss": second_avg_val_loss,
            }, checkpoint_target)
            statistics_target = save_directory / FileNameResolver.get_stats_filename(epoch + 1)
            print(f"Saving statistics to {statistics_target}")
            second_stats_dataframe.to_csv(statistics_target, index=False)

    # after loop
    print("=" * 40)

    # save best model
    if second_best_model_state_dict is None or second_best_optimizer_state_dict is None:
        print("No checkpoint found for best model!")
    else:
        best_checkpoint_target = save_directory / FileNameResolver.get_best_checkpoint_name(
            second_best_epoch + 1)
        torch.save({
            "epoch": second_best_epoch + 1,
            "model_state_dict": second_best_model_state_dict,
            "optimizer_state_dict": second_best_optimizer_state_dict,
            "train_loss": second_best_train_loss,
            "val_loss": second_best_val_loss,
        }, best_checkpoint_target)

        best_json_target = save_directory / FileNameResolver.get_best_json_filename(second_best_epoch + 1)
        with open(best_json_target, "w") as f:
            best_json = {
                "epoch": second_best_epoch + 1,
                "train_loss": second_best_train_loss,
                "val_loss": second_best_val_loss,
                "metrics": second_best_metrics
            }
            json.dump(best_json, f, indent=4)

    # save last checkpoint
    final_checkpoint_target = save_directory / FileNameResolver.get_final_checkpoint_name(
        second_stage_epochs)
    final_statistics_target = save_directory / FileNameResolver.get_final_stats_filename(
        second_stage_epochs)
    if save_checkpoints and second_stage_epochs % frequency == 0:
        # already saved in loop -> rename instead
        epoch_checkpoint_target = save_directory / FileNameResolver.get_checkpoint_name(
            second_stage_epochs)
        epoch_statistics_target = save_directory / FileNameResolver.get_stats_filename(second_stage_epochs)
        epoch_checkpoint_target.rename(final_checkpoint_target)
        epoch_statistics_target.rename(final_statistics_target)
    else:
        torch.save({
            "epoch": second_stage_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": second_stage_optimizer.state_dict(),
            "train_loss": second_avg_train_loss,
            "val_loss": second_avg_val_loss,
        }, final_checkpoint_target)
        first_stats_dataframe.to_csv(final_statistics_target, index=False)
    # json
    final_json_target = save_directory / FileNameResolver.get_final_json_filename(second_stage_epochs)
    with open(final_json_target, "w") as f:
        final_json = {
            "epoch": second_stage_epochs,
            "train_loss": second_avg_train_loss,
            "val_loss": second_avg_val_loss,
            "metrics": second_epoch_metrics
        }
        json.dump(final_json, f, indent=4)

    return (second_best_model_state_dict,
            second_best_optimizer_state_dict,
            second_best_train_loss,
            second_best_metrics,
            second_best_epoch,
            second_stats_dataframe)


def training_luna16_double(model: torch.nn.Module,
                           luna16_train_loader: torch.utils.data.DataLoader,
                           luna16_validate_loader: torch.utils.data.DataLoader,
                           luna25_train_loader: torch.utils.data.DataLoader,
                           luna25_validate_loader: torch.utils.data.DataLoader,
                           model_type: Literal["resnet3d", "medicalnet", "vivit"] = "resnet3d",
                           lr1: float = 5e-5,
                           lr2: float = 5e-5,
                           decay1: float = 1e-3,
                           decay2: float = 1e-3,
                           luna25_weight: float = 2.0,
                           metric: Literal[
                               "loss", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"] = "f1",
                           higher_is_better: bool = None,
                           threshold: float = 0.5,
                           forward_args: Mapping[str, Any] = None,
                           frequency: int = 10,
                           first_stage_epochs: int = 30,
                           second_stage_epochs: int = 30,
                           save_checkpoints: bool = True,
                           save_directory: str | Path = None,
                           device: torch.device | str = None, ):
    print(f"Model Name: {model.__class__.__module__} -> {model.__class__.__name__}")
    print(f"# Training on luna16 for {first_stage_epochs} epochs")
    print(f"# Training on luna25 for {second_stage_epochs} epochs")
    print("Initializing training loop...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Training will be run on {device}")

    if save_checkpoints:
        if save_directory is None:
            save_directory: Path = resolve_save_directory(model, save_directory)
        else:
            save_directory: Path = Path(save_directory).resolve()
        print(f"Saving weights and statistics to {save_directory} every {frequency} epochs")
        if not save_directory.is_dir():
            save_directory.mkdir(parents=True, exist_ok=True)

    if forward_args is None:
        forward_args = dict()

    # model_type = model_type.lower()
    vivit = "vivit" in model_type

    print()
    print(("#" * 20) + "LUNA16 TRAINING" + ("#" * 20))
    print()

    print("Initializing first stage training...")
    luna16_criterion = nn.BCEWithLogitsLoss().to(device)
    luna16_optimizer = AdamW(model.parameters(), lr=lr1, weight_decay=decay1)

    luna16_best_model_state_dict = None
    luna16_best_optimizer_state_dict = None
    luna16_best_train_loss = float("inf")
    luna16_best_val_loss = float("inf")
    luna16_best_metrics = None

    # resolve merics
    if higher_is_better is None:
        higher_is_better = metric != "loss"

    if higher_is_better:
        luna16_best_metric = float("-inf")
    else:
        # lower is better
        luna16_best_metric = float("inf")
    luna16_best_epoch = -1

    # dataframe to track training loss, validation loss, metrics
    luna16_stats_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "val_loss",
                                                  "accuracy", "precision", "recall", "f1", "auroc",
                                                  "average_precision"])

    # set average training and validation loss for ref
    luna16_avg_train_loss = None
    luna16_avg_val_loss = None
    luna16_epoch_metrics = None

    model.to(device)

    for epoch in tqdm(range(first_stage_epochs), desc="LUNA16 Progress"):
        model.train()
        train_loss = 0.0

        for inputs, labels in luna16_train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            luna16_optimizer.zero_grad()

            y_pred = model(inputs, **forward_args)
            if vivit:
                y_pred = y_pred.logits

            loss = luna16_criterion(y_pred, labels)
            loss.backward()
            luna16_optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        labels_tensor = torch.empty(0, dtype=torch.int).cpu()
        predictions_tensor = torch.empty(0).cpu()

        with torch.no_grad():
            for inputs, labels in luna16_validate_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                y_pred = model(inputs, **forward_args)
                if vivit:
                    y_pred = y_pred.logits
                loss = luna16_criterion(y_pred, labels)
                val_loss += loss.item()

                # append
                y_prob = torch.sigmoid(y_pred).cpu().reshape(-1)
                labels_tensor = torch.cat((labels_tensor, labels.cpu().int().reshape(-1)), 0)
                predictions_tensor = torch.cat((predictions_tensor, y_prob), 0)

        luna16_avg_train_loss = train_loss / len(luna16_train_loader)
        luna16_avg_val_loss = val_loss / len(luna16_validate_loader)

        luna16_epoch_metrics = run_metrics(y_prob=predictions_tensor,
                                          y_true=labels_tensor,
                                          threshold=threshold,
                                          skip_confusion_matrix=True)

        # report
        print("=" * 20)
        print(f"EPOCH: {epoch + 1}")
        print(f"Train Loss: {luna16_avg_train_loss}")
        print(f"Validation Loss: {luna16_avg_val_loss}")
        print(f"Accuracy: {luna16_epoch_metrics['accuracy']}")
        print(f"Precision: {luna16_epoch_metrics['precision']}")
        print(f"Recall: {luna16_epoch_metrics['recall']}")
        print(f"F1 Score: {luna16_epoch_metrics['f1']}")
        print(f"AUROC: {luna16_epoch_metrics['auroc']}")
        print(f"Average Precision: {luna16_epoch_metrics['average_precision']}")

        # update pd
        luna16_stats_dataframe.loc[len(luna16_stats_dataframe)] = [epoch + 1,
                                                                 luna16_avg_train_loss,
                                                                 luna16_avg_val_loss,
                                                                 luna16_epoch_metrics['accuracy'],
                                                                 luna16_epoch_metrics['precision'],
                                                                 luna16_epoch_metrics['recall'],
                                                                 luna16_epoch_metrics['f1'],
                                                                 luna16_epoch_metrics['auroc'],
                                                                 luna16_epoch_metrics['average_precision']]

        if metric == "loss":
            this_epoch_metric = luna16_avg_val_loss
        else:
            this_epoch_metric = luna16_epoch_metrics[metric]

        if higher_is_better:
            improvement = this_epoch_metric > luna16_best_metric
        else:
            # lower is better
            improvement = this_epoch_metric < luna16_best_metric
        if improvement:
            print(
                f"Model improved on validation metric: {metric}, saving best model state dict and optimizer state dict")
            luna16_best_val_loss = luna16_avg_val_loss
            luna16_best_model_state_dict = deepcopy(model.state_dict())
            luna16_best_optimizer_state_dict = deepcopy(luna16_optimizer.state_dict())
            luna16_best_train_loss = luna16_avg_train_loss
            luna16_best_epoch = epoch
            luna16_best_metric = this_epoch_metric
            luna16_best_metrics = luna16_epoch_metrics

        if save_checkpoints and (epoch + 1) % frequency == 0:
            checkpoint_target = save_directory / LUNA16FileNameResolver.get_checkpoint_name(epoch + 1)
            print(f"Saving checkpoint to {checkpoint_target}")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": luna16_optimizer.state_dict(),
                "train_loss": luna16_avg_train_loss,
                "val_loss": luna16_avg_val_loss,
            }, checkpoint_target)
            statistics_target = save_directory / LUNA16FileNameResolver.get_stats_filename(epoch + 1)
            print(f"Saving statistics to {statistics_target}")
            luna16_stats_dataframe.to_csv(statistics_target, index=False)

    # after loop
    print("=" * 40)

    # save best model
    if luna16_best_model_state_dict is None or luna16_best_optimizer_state_dict is None:
        print("No checkpoint found for best model!")
        raise RuntimeError("No checkpoint found for best model!")
    else:
        best_checkpoint_target = save_directory / LUNA16FileNameResolver.get_best_checkpoint_name(
            luna16_best_epoch + 1)
        torch.save({
            "epoch": luna16_best_epoch + 1,
            "model_state_dict": luna16_best_model_state_dict,
            "optimizer_state_dict": luna16_best_optimizer_state_dict,
            "train_loss": luna16_best_train_loss,
            "val_loss": luna16_best_val_loss,
        }, best_checkpoint_target)

        best_json_target = save_directory / LUNA16FileNameResolver.get_best_json_filename(luna16_best_epoch + 1)
        with open(best_json_target, "w") as f:
            best_json = {
                "epoch": luna16_best_epoch + 1,
                "train_loss": luna16_best_train_loss,
                "val_loss": luna16_best_val_loss,
                "metrics": luna16_best_metrics
            }
            json.dump(best_json, f, indent=4)

    # save last checkpoint
    final_checkpoint_target = save_directory / LUNA16FileNameResolver.get_final_checkpoint_name(
        first_stage_epochs)
    final_statistics_target = save_directory / LUNA16FileNameResolver.get_final_stats_filename(
        first_stage_epochs)
    if save_checkpoints and first_stage_epochs % frequency == 0:
        # already saved in loop -> rename instead
        epoch_checkpoint_target = save_directory / LUNA16FileNameResolver.get_checkpoint_name(
            first_stage_epochs)
        epoch_statistics_target = save_directory / LUNA16FileNameResolver.get_stats_filename(first_stage_epochs)
        epoch_checkpoint_target.rename(final_checkpoint_target)
        epoch_statistics_target.rename(final_statistics_target)
    else:
        torch.save({
            "epoch": first_stage_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": luna16_optimizer.state_dict(),
            "train_loss": luna16_avg_train_loss,
            "val_loss": luna16_avg_val_loss,
        }, final_checkpoint_target)
        luna16_stats_dataframe.to_csv(final_statistics_target, index=False)
    # json
    final_json_target = save_directory / LUNA16FileNameResolver.get_final_json_filename(first_stage_epochs)
    with open(final_json_target, "w") as f:
        final_json = {
            "epoch": first_stage_epochs,
            "train_loss": luna16_avg_train_loss,
            "val_loss": luna16_avg_val_loss,
            "metrics": luna16_epoch_metrics
        }
        json.dump(final_json, f, indent=4)

    # done
    print("LUNA16 finished")
    print("=" * 40)
    print()
    print(("#" * 20) + "LUNA25 FINETUNING" + ("#" * 20))
    print()
    # reporting
    print(f"Continuing from epoch: {luna16_best_epoch + 1}")
    print("Statistics:")
    print(f"Training Loss: {luna16_best_train_loss}")
    print(f"Validation Loss: {luna16_best_val_loss}")
    print(f"Metrics ({metric}): {luna16_best_metrics}")
    print("=" * 40)
    print("Initializing LUNA25 finetuning...")

    # reverting model to best epoch
    model.load_state_dict(luna16_best_model_state_dict)

    # optim
    luna25_optimizer = AdamW(model.parameters(),
                                   lr=lr2,
                                   weight_decay=decay2)

    # loss
    luna25_criterion = get_BCE_loss(pos_weight=luna25_weight).to(device)

    luna25_best_epoch = -1
    luna25_best_train_loss = luna16_best_train_loss
    luna25_best_val_loss = luna16_best_val_loss
    luna25_best_metric = luna16_best_metric
    luna25_best_metrics = luna16_best_metrics
    luna25_best_model_state_dict = deepcopy(luna16_best_model_state_dict)
    luna25_best_optimizer_state_dict = None

    # dataframe to track training loss, validation loss, metrics
    luna25_stats_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "val_loss",
                                                   "accuracy", "precision", "recall", "f1", "auroc",
                                                   "average_precision"])

    # set average training and validation loss for ref
    luna25_avg_train_loss = None
    luna25_avg_val_loss = None
    luna25_epoch_metrics = None

    model.to(device)

    for epoch in tqdm(range(second_stage_epochs), desc="LUNA25 Progress"):
        model.train()
        train_loss = 0.0

        for inputs, labels in luna25_train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            luna25_optimizer.zero_grad()

            y_pred = model(inputs, **forward_args)
            if vivit:
                y_pred = y_pred.logits

            loss = luna25_criterion(y_pred, labels)
            loss.backward()
            luna25_optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        labels_tensor = torch.empty(0, dtype=torch.int).cpu()
        predictions_tensor = torch.empty(0).cpu()

        with torch.no_grad():
            for inputs, labels in luna25_validate_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                y_pred = model(inputs, **forward_args)
                if vivit:
                    y_pred = y_pred.logits
                loss = luna16_criterion(y_pred, labels)
                val_loss += loss.item()

                # append
                y_prob = torch.sigmoid(y_pred).cpu().reshape(-1)
                labels_tensor = torch.cat((labels_tensor, labels.cpu().int().reshape(-1)), 0)
                predictions_tensor = torch.cat((predictions_tensor, y_prob), 0)

        luna25_avg_train_loss = train_loss / len(luna25_train_loader)
        luna25_avg_val_loss = val_loss / len(luna25_validate_loader)

        luna25_epoch_metrics = run_metrics(y_prob=predictions_tensor,
                                           y_true=labels_tensor,
                                           threshold=threshold,
                                           skip_confusion_matrix=True)

        # report
        print("=" * 20)
        print(f"EPOCH: {epoch + 1}")
        print(f"Train Loss: {luna25_avg_train_loss}")
        print(f"Validation Loss: {luna25_avg_val_loss}")
        print(f"Accuracy: {luna25_epoch_metrics['accuracy']}")
        print(f"Precision: {luna25_epoch_metrics['precision']}")
        print(f"Recall: {luna25_epoch_metrics['recall']}")
        print(f"F1 Score: {luna25_epoch_metrics['f1']}")
        print(f"AUROC: {luna25_epoch_metrics['auroc']}")
        print(f"Average Precision: {luna25_epoch_metrics['average_precision']}")

        # update pd
        luna25_stats_dataframe.loc[len(luna25_stats_dataframe)] = [epoch + 1,
                                                                   luna25_avg_train_loss,
                                                                   luna25_avg_val_loss,
                                                                   luna25_epoch_metrics['accuracy'],
                                                                   luna25_epoch_metrics['precision'],
                                                                   luna25_epoch_metrics['recall'],
                                                                   luna25_epoch_metrics['f1'],
                                                                   luna25_epoch_metrics['auroc'],
                                                                   luna25_epoch_metrics['average_precision']]

        if metric == "loss":
            this_epoch_metric = luna25_avg_val_loss
        else:
            this_epoch_metric = luna25_epoch_metrics[metric]

        if higher_is_better:
            improvement = this_epoch_metric > luna25_best_metric
        else:
            # lower is better
            improvement = this_epoch_metric < luna25_best_metric
        if improvement:
            print(
                f"Model improved on validation metric: {metric}, saving best model state dict and optimizer state dict")
            luna25_best_val_loss = luna25_avg_val_loss
            luna25_best_model_state_dict = deepcopy(model.state_dict())
            luna25_best_optimizer_state_dict = deepcopy(luna25_optimizer.state_dict())
            luna25_best_train_loss = luna25_avg_train_loss
            luna25_best_epoch = epoch
            luna25_best_metric = this_epoch_metric
            luna25_best_metrics = luna25_epoch_metrics

        if save_checkpoints and (epoch + 1) % frequency == 0:
            checkpoint_target = save_directory / FileNameResolver.get_checkpoint_name(epoch + 1)
            print(f"Saving checkpoint to {checkpoint_target}")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": luna25_optimizer.state_dict(),
                "train_loss": luna25_avg_train_loss,
                "val_loss": luna25_avg_val_loss,
            }, checkpoint_target)
            statistics_target = save_directory / FileNameResolver.get_stats_filename(epoch + 1)
            print(f"Saving statistics to {statistics_target}")
            luna25_stats_dataframe.to_csv(statistics_target, index=False)

    # after loop
    print("=" * 40)

    # save best model
    if luna25_best_model_state_dict is None or luna25_best_optimizer_state_dict is None:
        print("No checkpoint found for best model!")
    else:
        best_checkpoint_target = save_directory / FileNameResolver.get_best_checkpoint_name(
            luna25_best_epoch + 1)
        torch.save({
            "epoch": luna25_best_epoch + 1,
            "model_state_dict": luna25_best_model_state_dict,
            "optimizer_state_dict": luna25_best_optimizer_state_dict,
            "train_loss": luna25_best_train_loss,
            "val_loss": luna25_best_val_loss,
        }, best_checkpoint_target)

        best_json_target = save_directory / FileNameResolver.get_best_json_filename(luna25_best_epoch + 1)
        with open(best_json_target, "w") as f:
            best_json = {
                "epoch": luna25_best_epoch + 1,
                "train_loss": luna25_best_train_loss,
                "val_loss": luna25_best_val_loss,
                "metrics": luna25_best_metrics
            }
            json.dump(best_json, f, indent=4)

    # save last checkpoint
    final_checkpoint_target = save_directory / FileNameResolver.get_final_checkpoint_name(
        second_stage_epochs)
    final_statistics_target = save_directory / FileNameResolver.get_final_stats_filename(
        second_stage_epochs)
    if save_checkpoints and second_stage_epochs % frequency == 0:
        # already saved in loop -> rename instead
        epoch_checkpoint_target = save_directory / FileNameResolver.get_checkpoint_name(
            second_stage_epochs)
        epoch_statistics_target = save_directory / FileNameResolver.get_stats_filename(second_stage_epochs)
        epoch_checkpoint_target.rename(final_checkpoint_target)
        epoch_statistics_target.rename(final_statistics_target)
    else:
        torch.save({
            "epoch": second_stage_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": luna25_optimizer.state_dict(),
            "train_loss": luna25_avg_train_loss,
            "val_loss": luna25_avg_val_loss,
        }, final_checkpoint_target)
        luna16_stats_dataframe.to_csv(final_statistics_target, index=False)
    # json
    final_json_target = save_directory / FileNameResolver.get_final_json_filename(second_stage_epochs)
    with open(final_json_target, "w") as f:
        final_json = {
            "epoch": second_stage_epochs,
            "train_loss": luna25_avg_train_loss,
            "val_loss": luna25_avg_val_loss,
            "metrics": luna25_epoch_metrics
        }
        json.dump(final_json, f, indent=4)

    return (luna25_best_model_state_dict,
            luna25_best_optimizer_state_dict,
            luna25_best_train_loss,
            luna25_best_metrics,
            luna25_best_epoch,
            luna25_stats_dataframe)
