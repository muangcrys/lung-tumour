from pathlib import Path

class PathList:
    # project root in reference to this file
    project_root = Path(__file__).resolve().parent.parent.parent

    # config path
    config_dir = project_root / "config"

    # data
    data_dir = project_root / "data"

    # luna data
    luna_data_dir = data_dir / "LUNA"
    raw_luna_data_dir = luna_data_dir / "raw"
    processed_luna_data_dir = luna_data_dir / "processed"

    # raw luna annotation
    raw_luna_annotation_dir = raw_luna_data_dir / "annotation"
    raw_luna_annotation_csv = raw_luna_annotation_dir / "LUNA25_Public_Training_Development_Data.csv"

    # raw luna image
    raw_luna_image_dir = raw_luna_data_dir / "image"

    # intermediate processing
    intermediate_luna_data_dir = luna_data_dir / "intermediate"

    # patients
    patients_csv = intermediate_luna_data_dir / "patients.csv"

    # annotations (4242)
    train_annotation_csv = processed_luna_data_dir / "SEED_4242" / "train_annotations.csv"
    validate_annotation_csv = processed_luna_data_dir / "SEED_4242" / "validate_annotations.csv"
    test_annotation_csv = processed_luna_data_dir / "SEED_4242" / "test_annotations.csv"

    # kfold annotations
    k_fold_annotation_dir = processed_luna_data_dir / "k_fold_annotations"

    # trained checkpoints
    saved_weights_dir = project_root / "weights"
    saved_2stage_weights_dir = project_root / "weights_2stage"
    saved_kfold_weights_dir = project_root / "weights_kfold"
    saved_kfold_2stage_weights_dir = project_root / "weights_kfold_2stage"

    # outputs
    output_dir = project_root / "output"

    # kfold metric
    k_fold_output_dir = output_dir / "k_fold_output.csv"
    k_fold_2stage_output_dir = output_dir / "k_fold_2stage_output.csv"



    