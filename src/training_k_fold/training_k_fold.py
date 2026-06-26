from pathlib import Path

from matplotlib.testing.jpl_units import km

from utility.reproducibility import k_fold_seed, DEFAULT_SEED
from typing import Literal
from utility.utils import get_timestamp_now
from utility.paths import PathList
from training.train_fresh_medicalnet import train_fresh_medicalnet
from training.train_fresh_resnet3d import train_fresh_resnet3d
from training.train_medicalnet import train_medicalnet
from training.train_resnet3d import train_resnet3d
from training.train_vivit import train_vivit


def k_fold_validate_annotation_csv(k: int) -> str:
    return f"fold_{k}_validate.csv"


def k_fold_train_annotation_csv(k: int) -> str:
    return f"fold_{k}_train.csv"


def k_fold_train_wrapper(
        model_string: Literal["resnet3d", "medicalnet", "fresh_resnet3d", "fresh_medicalnet"],
        folds: int = 4,
        seed: int = DEFAULT_SEED,
        fold_annotation_dir: str | Path | None = None,

        # intercepting arguments
        train_annotation: str | Path | None = None,
        validate_annotation: str | Path | None = None,

        **kwargs
):
    # timestamp for this run
    run_timestamp = get_timestamp_now()

    # resolve annotation directory first
    if fold_annotation_dir is None:
        fold_annotation_dir = PathList.k_fold_annotation_dir.resolve()
    else:
        fold_annotation_dir = Path(fold_annotation_dir).resolve()
    model_string = model_string.lower()
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
                             k=k, )

        # get train and validate annotation paths
        train_annotation = fold_annotation_dir / k_fold_train_annotation_csv(k=k)
        validate_annotation = fold_annotation_dir / k_fold_validate_annotation_csv(k=k)

        print(
            "###################################################################################################################################")
        print(
            f"###                                                      FOLD {k}                                                                 ###")
        print(
            "###################################################################################################################################")

        # pass to model function
        training_fn(
            train_annotation=train_annotation,
            validate_annotation=validate_annotation,
            seed=k_seed,
            k=k,
            time_stamp=run_timestamp,
            **kwargs
        )


def k_fold_vivit_wrapper(
        k: int = 1,
        seed: int = DEFAULT_SEED,
        fold_annotation_dir: str | Path | None = None,
        pretrained: bool = True,

        # intercepting arguments
        train_annotation: str | Path | None = None,
        validate_annotation: str | Path | None = None,
        mpdel_string: str | None = None,  # just in case

        **kwargs
):
    # since we only have 4 folds of data for now just assert it
    assert 0 < k <= 4

    run_timestamp = get_timestamp_now()

    if fold_annotation_dir is None:
        fold_annotation_dir = PathList.k_fold_annotation_dir.resolve()
    else:
        fold_annotation_dir = Path(fold_annotation_dir).resolve()

    # get train and validate annotation paths
    train_annotation = fold_annotation_dir / k_fold_train_annotation_csv(k=k)
    validate_annotation = fold_annotation_dir / k_fold_validate_annotation_csv(k=k)

    k_seed = k_fold_seed(initial_seed=seed,
                         k=k)


    print(
        "###################################################################################################################################")
    print(
        f"###                                                      FOLD {k}                                                                 ###")
    print(
        "###################################################################################################################################")

    # pass
    print(f"You are training a ViVit model which was {pretrained if pretrained else "randomly initialised"}")
    train_vivit(
        train_annotation=train_annotation,
        validate_annotation=validate_annotation,
        seed=k_seed,
        k=k,
        time_stamp=run_timestamp,
        **kwargs)
