from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd
from utility.reproducibility import DEFAULT_SEED, reset_seed
from utility.paths import PathList


def split_patients(patients_csv: str,
                   stratify_by: str = "stratifying_key",
                   test_size: float = 0.2,
                   seed: int = DEFAULT_SEED,
                   save_file: bool = True,
                   output_dir: str | Path = None):
    patients_df = pd.read_csv(patients_csv)
    dev_df, test_df = train_test_split(patients_df,
                                         test_size=test_size,
                                         stratify=patients_df[stratify_by],
                                         random_state=seed
                                         )

    if not save_file:
        return dev_df, test_df

    if output_dir is None:
        seed_dir = "SEED_" + str(seed)
        output_dir = PathList.intermediate_luna_data_dir / seed_dir

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_csv = output_dir / "dev_patients.csv"
    test_csv = output_dir / "test_patients.csv"

    dev_df.to_csv(dev_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return dev_df, test_df

def split_train_patients(patients_train_csv: str|Path|pd.DataFrame,
                         stratify_by: str = "stratifying_key",
                         validate_size: float = 0.25,
                         seed: int = DEFAULT_SEED,
                         save_file: bool = True,
                         output_dir: str | Path = None):
    if isinstance(patients_train_csv, pd.DataFrame):
        patients_train_df = patients_train_csv
    else:
        patients_train_csv = Path(patients_train_csv)
        patients_train_df = pd.read_csv(patients_train_csv)

    train_df, validation_df = train_test_split(patients_train_df,
                                               test_size=validate_size,
                                               stratify=patients_train_df[stratify_by],
                                               random_state=seed)
    if not save_file:
        return train_df, validation_df
    if output_dir is None:
        seed_dir = "SEED_" + str(seed)
        output_dir = PathList.intermediate_luna_data_dir / seed_dir

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train_patients.csv"
    validate_csv = output_dir / "validate_patients.csv"

    train_df.to_csv(train_csv, index=False)
    validation_df.to_csv(validate_csv, index=False)
    return train_df, validation_df


def fetch_image(patients_csv: str | Path,
                annotations_csv: str | Path,
                save_to: str | Path = None,
                save_file: bool = True):
    split_patients_df = pd.read_csv(patients_csv)
    annotations_df = pd.read_csv(annotations_csv)

    # select only rows with PatientID in split_patients_df
    annotation_in_split_df = annotations_df[annotations_df["PatientID"].isin(split_patients_df["PatientID"])]

    if save_file:
        annotation_in_split_df.to_csv(save_to, index=False)

    return annotation_in_split_df


def split_image(annotations_csv: str | Path,
                train_csv: str | Path = None,
                validate_csv: str | Path = None,
                test_csv: str | Path = None,
                save_dir: str | Path = None, ):
    if train_csv is None and test_csv is None and validate_csv is None:
        print("No files are supplied.")
        return
    annotations_csv = Path(annotations_csv)
    train_csv: Path | None = Path(train_csv) if train_csv is not None else None
    validate_csv: Path | None = Path(validate_csv) if validate_csv is not None else None
    test_csv: Path | None = Path(test_csv) if test_csv is not None else None

    # resolve save dir
    if save_dir is None:
        if train_csv is not None:
            save_dir = train_csv.parent
        elif validate_csv is not None:
            save_dir = validate_csv.parent
        else:
            save_dir = test_csv.parent

    if train_csv is not None:
        # save location
        output_train_csv = save_dir / "train_annotations.csv"
        train_out = fetch_image(patients_csv=train_csv,
                                annotations_csv=annotations_csv,
                                save_to=output_train_csv,
                                save_file=True)
    else:
        train_out = None

    if validate_csv is not None:
        # save location
        output_validate_csv = save_dir / "validate_annotations.csv"
        validate_out = fetch_image(patients_csv=validate_csv,
                                   annotations_csv=annotations_csv,
                                   save_to=output_validate_csv,
                                   save_file=True)
    else:
        validate_out = None

    if test_csv is not None:
        # save location
        output_test_csv = save_dir / "test_annotations.csv"
        test_out = fetch_image(patients_csv=test_csv,
                               annotations_csv=annotations_csv,
                               save_to=output_test_csv,
                               save_file=True)
    else:
        test_out = None

    return train_out, validate_out, test_out
