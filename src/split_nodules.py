import argparse
from utility.paths import PathList
from utility.reproducibility import DEFAULT_SEED
from luna_dataset.splitting import split_image
from pathlib import Path
import pandas as pd


def report_nodule_characteristics(df: pd.DataFrame):
    nodules = df.shape[0]
    males = df[df["Gender"] == "Male"].shape[0]
    females = df[df["Gender"] == "Female"].shape[0]
    positive = df[df["label"] == 1].shape[0]
    negative = df[df["label"] == 0].shape[0]

    print(f"Total Nodules: {nodules}")
    print(f"M: {males}, F: {females} --> Ratio: {males / nodules} to {females / nodules}")
    print(f"Positive: {positive}, Negative: {negative} --> Ratio: {positive / negative}")


def main():
    default_seed_dir = "SEED_" + str(DEFAULT_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_csv", type=str, required=False,
                        default=PathList.raw_luna_annotation_csv)
    parser.add_argument("--patients_csv_dir", type=str, required=False,
                        default=PathList.intermediate_luna_data_dir / default_seed_dir)
    parser.add_argument("--save_to", type=str, required=False)

    args = parser.parse_args()
    patients_train_csv = Path(args.patients_csv_dir) / "train_patients.csv"
    patients_validate_csv = Path(args.patients_csv_dir) / "validate_patients.csv"
    patients_test_csv = Path(args.patients_csv_dir) / "test_patients.csv"
    train_df, validate_df, test_df = split_image(annotations_csv=args.annotation_csv,
                                                 train_csv=patients_train_csv,
                                                 validate_csv=patients_validate_csv,
                                                 test_csv=patients_test_csv,
                                                 save_dir=args.save_to,
                                                 )

    # report
    print("=" * 30)
    print("Train Set Characteristics (Nodules):")
    report_nodule_characteristics(train_df)
    print("=" * 30)
    print("Validate Set Characteristics (Nodules):")
    report_nodule_characteristics(validate_df)
    print("=" * 30)
    print("Test Set Characteristics (Nodules):")
    report_nodule_characteristics(test_df)

    # check intersection
    print("=" * 30)
    intersecting_nodules_df = train_df[train_df["AnnotationID"].isin(test_df["AnnotationID"])]
    print("Number of intersecting nodules: ", intersecting_nodules_df.shape[0])
    intersecting_patients_df = train_df[train_df["PatientID"].isin(test_df["PatientID"])]
    print("Number of nodules with intersecting patientID: ", intersecting_patients_df.shape[0])


if __name__ == "__main__":
    main()
