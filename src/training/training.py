import torch
from torch import nn
from torch.optim import AdamW
from typing import Literal, Mapping, Any
from pathlib import Path
from datetime import datetime
from evaluate.metrics import run_metrics
from training.names import FileNameResolver
from utility.paths import PathList
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
                           base_directory: str|Path = None,
                           model_string: str = None):
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_string is None:
        model_string = f"{model.__class__.__module__}-{model.__class__.__name__}"

    if base_directory is None:
        base_directory = PathList.saved_weights_dir.resolve()
    else:
        base_directory = Path(base_directory).resolve()

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
                  device: torch.device | str = None):


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
            save_directory:Path = resolve_save_directory(model, save_directory)
        else:
            save_directory:Path = Path(save_directory).resolve()
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
                                    skip_confusion_matrix = True)

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
            print("Validation loss improved from best validation loss!")
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
            "epoch": best_epoch,
            "model_state_dict": best_model_state_dict,
            "optimizer_state_dict": best_optimizer_state_dict,
            "train_loss": best_train_loss,
            "val_loss": best_val_loss,
        }, best_checkpoint_target)

        best_json_target = save_directory / FileNameResolver.get_best_json_filename(best_epoch + 1)
        with open(best_json_target, "w") as f:
            best_json = {
                "epoch": best_epoch,
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


