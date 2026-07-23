import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_loss_curves_from_df(df: pd.DataFrame,
                             figsize: tuple[int, int] = (7, 7),
                             best_epoch: int = None,
                             show_plt: bool = False):
    # melt df into long format
    df_long = df.melt(id_vars=["epoch"],
                      value_vars=["train_loss", "val_loss"],
                      var_name="loss_type",
                      value_name="loss")

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df_long,
        x="epoch",
        y="loss",
        hue="loss_type",
        ax=ax
    )

    if best_epoch is not None:
        # plot a vertical line at the best epoch
        ax.axvline(x=best_epoch, color="red", linestyle="--", label="Best Epoch")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves")
    ax.legend()
    if show_plt:
        plt.show()
    return fig, ax


def plot_loss_curves_from_csv(csv_path: str | Path,
                              figsize: tuple[int, int] = (7, 7),
                              save_plot: bool = True,
                              save_directory: str | Path = None,
                              save_prefix: str | None = None,
                              best_epoch: int = None,
                              file_name="loss_curves.pdf"):
    # check that csv exists and can be loaded
    csv_path: Path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} is not a file")
    df = pd.read_csv(csv_path)
    # resolve save
    if save_plot:
        if save_directory is None:
            save_directory = csv_path.parent.resolve()
        else:
            save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        if save_prefix is not None:
            file_name = f"{save_prefix}_{file_name}"

    # plot
    fig, ax = plot_loss_curves_from_df(df, figsize=figsize, show_plt=False, best_epoch=best_epoch)

    if save_plot:
        fig.savefig(save_directory / file_name, bbox_inches="tight")
    return fig, ax


def plot_2_stage_loss_curves_from_df(df1: pd.DataFrame,
                                     df2: pd.DataFrame,
                                     first_stage_stop: int,
                                     figsize: tuple[int, int] = (7, 7),
                                     best_epoch: int = None,
                                     show_plt: bool = False, ):
    max_stage1 = first_stage_stop

    # crop df1
    df1 = df1.copy()
    df1 = df1[df1["epoch"] <= max_stage1]

    # shift df2
    df2 = df2.copy()
    df2["epoch"] = df2["epoch"] + max_stage1

    # melt
    df_long1 = df1.melt(id_vars=["epoch"],
                        value_vars=["train_loss", "val_loss"],
                        var_name="loss_type",
                        value_name="loss")
    df_long2 = df2.melt(id_vars=["epoch"],
                        value_vars=["train_loss", "val_loss"],
                        var_name="loss_type",
                        value_name="loss")
    df_long = pd.concat([df_long1, df_long2], ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df_long,
        x="epoch",
        y="loss",
        hue="loss_type",
        ax=ax
    )

    if best_epoch is not None:
        ax.axvline(x=best_epoch + max_stage1, color="red", linestyle="--", label="Best Epoch")

    # vertical line between stages
    ax.axvline(x=max_stage1, color="black", linestyle="--", label=f"FT Starts (epoch {max_stage1}")

    # set legends
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves")
    ax.legend()

    if show_plt:
        plt.show()
    return fig, ax


def plot_2_stage_loss_curves_from_csv(csv_path1: str | Path,
                                      csv_path2: str | Path,
                                      first_stage_stop: int,
                                      figsize: tuple[int, int] = (7, 7),
                                      save_plot: bool = True,
                                      save_directory: str | Path = None,
                                      best_epoch: int = None,
                                      file_name="loss_curves.pdf"):
    # check that csv exists and can be loaded
    csv_path1: Path = Path(csv_path1)
    if not csv_path1.is_file():
        raise FileNotFoundError(f"{csv_path1} is not a file")
    df1 = pd.read_csv(csv_path1)

    csv_path2: Path = Path(csv_path2)
    if not csv_path2.is_file():
        raise FileNotFoundError(f"{csv_path2} is not a file")
    df2 = pd.read_csv(csv_path2)

    # resolve save
    if save_plot:
        if save_directory is None:
            save_directory = csv_path1.parent.resolve()
        else:
            save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

    # plot
    fig, ax = plot_2_stage_loss_curves_from_df(df1,
                                               df2,
                                               first_stage_stop,
                                               figsize=figsize,
                                               show_plt=False,
                                               best_epoch=best_epoch)

    if save_plot:
        fig.savefig(save_directory / file_name, bbox_inches="tight")
    return fig, ax
