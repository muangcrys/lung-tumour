import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_loss_curves_from_df(df: pd.DataFrame,
                             figsize: tuple[int, int] = (7, 7),
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
        hue="split",
        ax=ax
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves")
    if show_plt:
        plt.show()
    return fig, ax


def plot_loss_curves_from_csv(csv_path: str | Path,
                              figsize: tuple[int, int] = (7, 7),
                              save_plot: bool = True,
                              save_directory: str | Path = None,
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

    # plot
    fig, ax = plot_loss_curves_from_df(df, figsize=figsize, show_plt=False)

    if save_plot:
        fig.savefig(save_directory / file_name, bbox_inches = "tight")
    return fig, ax



