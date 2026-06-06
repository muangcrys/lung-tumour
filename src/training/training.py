import torch
from torch import nn
from torch.optim import AdamW
from typing import Literal, Mapping, Any
from pathlib import Path
from datetime import datetime
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
    best_epoch = -1

    # dataframe to track training and validation loss
    stats_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    # set average training and validation loss for ref
    avg_train_loss = None
    avg_val_loss = None

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

        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                y_pred = model(inputs, **forward_args)

                loss = criterion(y_pred, labels)

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(validate_loader)

        # check if this is the best model so far
        print("=" * 20)
        print(f"EPOCH: {epoch + 1}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Validation Loss: {avg_val_loss}")

        # update pd
        stats_dataframe.loc[len(stats_dataframe)] = [epoch + 1, avg_train_loss, avg_val_loss]

        if best_val_loss > avg_val_loss:
            print("Validation loss improved from best validation loss!")
            best_val_loss = avg_val_loss
            best_model_state_dict = deepcopy(model.state_dict())
            best_optimizer_state_dict = deepcopy(optimizer.state_dict())
            best_train_loss = avg_train_loss
            best_epoch = epoch

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
        }
        json.dump(final_json, f, indent=4)

    # done
    print("=" * 40)
    print("Done!")

    return best_model_state_dict, best_optimizer_state_dict, best_train_loss, best_val_loss, best_epoch, stats_dataframe


class FileNameResolver:
    @staticmethod
    def get_checkpoint_name(epoch: int) -> str:
        epoch_str = str(epoch)
        return epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_stats_filename(epoch: int) -> str:
        epoch_str = str(epoch)
        return epoch_str + "_" + "stats.csv"

    @staticmethod
    def get_best_checkpoint_name(epoch: int) -> str:
        epoch_str = str(epoch)
        return "#BEST" + epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_best_json_filename(epoch: int) -> str:
        epoch_str = str(epoch)
        return "#BEST_" + epoch_str + "_" + "details.json"

    @staticmethod
    def get_final_checkpoint_name(epoch: int) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_final_json_filename(epoch: int) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "details.json"

    @staticmethod
    def get_final_stats_filename(epoch: int) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "stats.csv"

    @staticmethod
    def get_training_configs_filename() -> str:
        return "###TrainingConfigs.json"


