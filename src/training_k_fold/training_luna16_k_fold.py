from training_k_fold.training_k_fold import k_fold_train_annotation_csv, k_fold_validate_annotation_csv
from typing import Literal
from utility.reproducibility import DEFAULT_SEED, k_fold_seed
from utility.paths import PathList
from pathlib import Path
from utility.utils import get_timestamp_now
from training.trainluna16_pretrained import train_luna16_pretrained

def k_fold_luna16_double_wrapper(
        model_string: Literal["resnet3d_pretrained", "medicalnet_pretrained"],
        folds: int = 4,
        seed: int = DEFAULT_SEED,
        luna16_fold_annotation_dir: str | Path | None = None,
        luna25_fold_annotation_dir: str | Path | None = None,

        # intercepting arguments
        train_annotation: str | Path | None = None,
        validate_annotation: str | Path | None = None,

        **kwargs):
    # timestamp for this run
    run_timestamp = get_timestamp_now()

    # resolve annotation directory first
    if luna16_fold_annotation_dir is None:
        luna16_fold_annotation_dir = PathList.luna16_kfold_annotation_dir.resolve()
    else:
        luna16_fold_annotation_dir = Path(luna16_fold_annotation_dir).resolve()
    if luna25_fold_annotation_dir is None:
        luna25_fold_annotation_dir = PathList.k_fold_annotation_dir.resolve()
    else:
        luna25_fold_annotation_dir = Path(luna25_fold_annotation_dir).resolve()

    model_string = model_string.lower()

    for k in range(1, folds + 1):
        # get the seed
        k_seed = k_fold_seed(initial_seed=seed,
                             k=k,)

        # get train and validate paths
        luna16_train_annotation = luna16_fold_annotation_dir / k_fold_train_annotation_csv(k=k)
        luna16_validate_annotation = luna16_fold_annotation_dir / k_fold_validate_annotation_csv(k=k)
        luna25_train_annotation = luna25_fold_annotation_dir / k_fold_train_annotation_csv(k=k)
        luna25_validate_annotation = luna25_fold_annotation_dir / k_fold_validate_annotation_csv(k=k)

        print(
            "###################################################################################################################################")
        print(
            f"###                                                      FOLD {k}                                                                 ###")
        print(
            "###################################################################################################################################")

        if model_string == "resnet3d_pretrained":
            train_luna16_pretrained(
                model_type="resnet3d",
                preprocessing_pipeline="video_pretrained",
                n_input_channels=3,
                luna16_train_annotation=luna16_train_annotation,
                luna16_validate_annotation=luna16_validate_annotation,
                luna25_train_annotation=luna25_train_annotation,
                luna25_validate_annotation=luna25_validate_annotation,
                seed=k_seed,
                k=k,
                time_stamp=run_timestamp,
                **kwargs
            )

        elif model_string == "medicalnet_pretrained":
            train_luna16_pretrained(
                model_type="medicalnet",
                preprocessing_pipeline="medical_pretrained",
                n_input_channels=1,
                luna16_train_annotation=luna16_train_annotation,
                luna16_validate_annotation=luna16_validate_annotation,
                luna25_train_annotation=luna25_train_annotation,
                luna25_validate_annotation=luna25_validate_annotation,
                seed=k_seed,
                k=k,
                time_stamp=run_timestamp,
                **kwargs
            )
        else:
            raise NotImplementedError()

def k_fold_luna16_double_vivit_wrapper(
        model_string: Literal["vivit_pretrained"] = "vivit_pretrained",
        k: int = 1,
        seed: int = DEFAULT_SEED,
        time_stamp: str | None = None,

        luna16_fold_annotation_dir: str | Path | None = None,
        luna25_fold_annotation_dir: str | Path | None = None,

        # intercepting arguments
        train_annotation: str | Path | None = None,
        validate_annotation: str | Path | None = None,

        **kwargs):
    assert 0 < k <= 4
    run_timestamp = time_stamp if time_stamp is not None else get_timestamp_now()
    if luna16_fold_annotation_dir is None:
        luna16_fold_annotation_dir = PathList.luna16_kfold_annotation_dir.resolve()
    else:
        luna16_fold_annotation_dir = Path(luna16_fold_annotation_dir).resolve()
    if luna25_fold_annotation_dir is None:
        luna25_fold_annotation_dir = PathList.k_fold_annotation_dir.resolve()
    else:
        luna25_fold_annotation_dir = Path(luna25_fold_annotation_dir).resolve()

    # get the seed
    k_seed = k_fold_seed(initial_seed=seed,
                         k=k, )

    # get train and validate paths
    luna16_train_annotation = luna16_fold_annotation_dir / k_fold_train_annotation_csv(k=k)
    luna16_validate_annotation = luna16_fold_annotation_dir / k_fold_validate_annotation_csv(k=k)
    luna25_train_annotation = luna25_fold_annotation_dir / k_fold_train_annotation_csv(k=k)
    luna25_validate_annotation = luna25_fold_annotation_dir / k_fold_validate_annotation_csv(k=k)

    print(
        "###################################################################################################################################")
    print(
        f"###                                                      FOLD {k}                                                                 ###")
    print(
        "###################################################################################################################################")

    train_luna16_pretrained(
        model_type="vivit",
        preprocessing_pipeline="vivit_pretrained",
        n_input_channels=3,
        luna16_train_annotation=luna16_train_annotation,
        luna16_validate_annotation=luna16_validate_annotation,
        luna25_train_annotation=luna25_train_annotation,
        luna25_validate_annotation=luna25_validate_annotation,
        seed=k_seed,
        k=k,
        time_stamp=run_timestamp,
        **kwargs
    )