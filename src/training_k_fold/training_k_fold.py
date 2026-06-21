from pathlib import Path
from utility.reproducibility import k_fold_seed, DEFAULT_SEED
from typing import Literal
from utility.utils import get_timestamp_now
from utility.paths import PathList
from training.train_fresh_medicalnet import train_fresh_medicalnet
from training.train_fresh_resnet3d import train_fresh_resnet3d
from training.train_medicalnet import train_medicalnet
from training.train_resnet3d import train_resnet3d


def k_fold_validate_annotation_csv(k: int) -> str:
    return f"fold_{k}_validate.csv"


def k_fold_train_annotation_csv(k: int) -> str:
    return f"fold_{k}_train.csv"

def k_fold_train_wrapper(
        model_string: Literal["resnet3d", "medicalnet", "fresh_resnet3d", "fresh_medicalnet"],
        folds: int = 4,
        seed: int = DEFAULT_SEED,
        fold_annotation_dir: str | Path | None = None,
        **kwargs
):
    # timestamp for this run
    run_timestamp = get_timestamp_now()

    # resolve annotation directory first
    if fold_annotation_dir is None:
        fold_annotation_dir = PathList.k_fold_annotation_dir.resolve()

    # resolve model training function
    if model_string == "resnet3d":
        training_fn = train_resnet3d
    elif model_string == "medicalnet":
        training_fn = train_medicalnet
    elif model_string == "fresh_resnet3d":
        training_fn = train_fresh_resnet3d
    elif model_string == "fresh_medicalnet":
        training_fn = train_fresh_medicalnet
    else:
        raise ValueError(f"Unknown model string: {model_string}")

    for k in range(1, folds + 1):
        # get the seed
        k_seed = k_fold_seed(initial_seed=seed,
                             k=k,)

        # get train and validate annotation paths
        train_annotation = fold_annotation_dir / k_fold_train_annotation_csv(k=k)
        validate_annotation = fold_annotation_dir / k_fold_validate_annotation_csv(k=k)

        # pass to model function
        training_fn(
            train_annotation=train_annotation,
            validate_annotation=validate_annotation,
            seed=k_seed,
            k=k,
            time_stamp=run_timestamp,
            **kwargs
        )



