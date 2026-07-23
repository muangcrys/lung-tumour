from training_k_fold.training_k_fold import k_fold_train_annotation_csv, k_fold_validate_annotation_csv
from typing import Literal
from utility.reproducibility import DEFAULT_SEED, k_fold_seed
from utility.paths import PathList
from pathlib import Path
from utility.utils import get_timestamp_now
from training.train2stage_medicalnet import train2stage_medicalnet
from training.train2stage_resnet3d import train2stage_resnet3d
from training.train2stage_vivit import train2stage_vivit

def k_fold_2_stage_wrapper(
    model_string: Literal["resnet3d_2stage", "medicalnet_2stage", "vivit_2stage"],
    folds: int = 4,
    seed: int = DEFAULT_SEED,
    fold_annotation_dir: str | Path | None = None,

    # intercepting arguments
    train_annotation: str | Path | None = None,
    validate_annotation: str | Path | None = None,

    **kwargs):

    # timestamp for this run
    run_timestamp = get_timestamp_now()

    # resolve annotation directory first
    if fold_annotation_dir is None:
        fold_annotation_dir = PathList.k_fold_annotation_dir.resolve()
    else:
        fold_annotation_dir = Path(fold_annotation_dir).resolve()
    model_string = model_string.lower()
    # resolve model training function
    if model_string == "resnet3d_2stage":
        training_fn = train2stage_resnet3d
    elif model_string == "medicalnet_2stage":
        training_fn = train2stage_medicalnet
    elif model_string == "vivit_2stage":
        training_fn = train2stage_vivit
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

def k_fold_2_stage_vivit_wrapper(
        k: int = 1,
        seed: int = DEFAULT_SEED,
        fold_annotation_dir: str | Path | None = None,

        # intercepting arguments
        train_annotation: str | Path | None = None,
        validate_annotation: str | Path | None = None,
        time_stamp: str | None = None,

        **kwargs
):
    # since we only have 4 folds of data for now just assert it
    assert 0 < k <= 4

    run_timestamp = time_stamp if time_stamp is not None else get_timestamp_now()

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
    train2stage_vivit(
        train_annotation=train_annotation,
        validate_annotation=validate_annotation,
        seed=k_seed,
        k=k,
        time_stamp=run_timestamp,
        **kwargs)
