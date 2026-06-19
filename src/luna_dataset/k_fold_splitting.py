from pathlib import Path

from sklearn.model_selection import StratifiedKFold
import pandas as pd
from utility.reproducibility import DEFAULT_SEED, reset_seed
from utility.paths import PathList
from luna_dataset.splitting import fetch_image


def k_fold_split_patients(patients_csv: str,
                          stratify_by: str = "stratifying_key",
                          k: int = 4,
                          seed: int = DEFAULT_SEED,
                          save_file: bool = True,
                          output_dir: str | Path = None):
    patients_df = pd.read_csv(patients_csv)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    stratify = patients_df[stratify_by].copy()
    trains = []
    vals = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(patients_df, stratify)):
        train_df = patients_df.iloc[train_idx]
        val_df = patients_df.iloc[val_idx]

        trains.append(train_df)
        vals.append(val_df)

    if not save_file:
        return trains, vals

    if output_dir is None:
        output_dir = Path(patients_csv).parent / "k_fold_patients"

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold, train_df in enumerate(trains, start=1):
        target = output_dir / f"fold_{fold}_train.csv"
        train_df.to_csv(target, index=False)

    for fold, val_df in enumerate(vals, start=1):
        target = output_dir / f"fold_{fold}_validate.csv"
        val_df.to_csv(target, index=False)

    return trains, vals


def k_fold_split_image(annotations_csv: str | Path,
                       fold_patients_dir: str | Path,
                       save_dir: str | Path = None):
    annotations_csv = Path(annotations_csv)
    fold_patients_dir = Path(fold_patients_dir)

    # resolve save dir
    if save_dir is None:
        # resolve save dir based on fold_patients_dir
        save_dir = fold_patients_dir.parent / "k_fold_annotations"
        save_dir.mkdir(parents=True, exist_ok=True)
    # get all files in fold_patients_dir
    file_list = fold_patients_dir.glob("fold_*_*.csv")
    trains = []
    vals = []
    for csv in file_list:
        # csv is Path object
        filename = csv.name
        filestem = csv.stem
        split = filestem.split("_")[2]

        # forward to save_dir
        target = save_dir / filename

        df = fetch_image(patients_csv=csv,
                         annotations_csv=annotations_csv,
                         save_to=target,
                         save_file=True,)
        if split == "train":
            trains.append(df)
        elif split == "validate":
            vals.append(df)
        else:
            print(f"Unknown split: {split}")

    return trains, vals



