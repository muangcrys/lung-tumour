import argparse

from luna_dataset.k_fold_splitting import k_fold_split_patients, k_fold_split_image
from utility.paths import PathList
from utility.reproducibility import DEFAULT_SEED
from luna_dataset.splitting import split_image
from pathlib import Path
import pandas as pd
from split_nodules import report_nodule_characteristics


def main():
    default_seed_dir = "SEED_" + str(DEFAULT_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients_csv", type=str, required=False,
                        default=PathList.intermediate_luna_data_dir / default_seed_dir / "dev_patients.csv")
    parser.add_argument("--stratify_by", type=str, required=False, default="stratifying_key")
    parser.add_argument("--k", type=int, required=False, default=4)
    parser.add_argument("--seed", type=int, required=False, default=DEFAULT_SEED)
    parser.add_argument("--output_dir", type=str, required=False)

    parser.add_argument("--annotations_csv", type=str, required=False, default=PathList.raw_luna_annotation_csv)

    args = parser.parse_args()

    k_fold_patients_location = Path(args.output_dir).resolve().parent / "k_fold_patients" if args.output_dir \
        else Path(args.patients_csv).resolve().parent / "k_fold_patients"

    _ = k_fold_split_patients(patients_csv=args.patients_csv,
                              stratify_by=args.stratify_by,
                              k=args.k,
                              seed=args.seed,
                              output_dir=k_fold_patients_location)

    trains, vals = k_fold_split_image(annotations_csv=args.annotations_csv,
                                      fold_patients_dir=k_fold_patients_location,
                                      save_dir=args.output_dir, )

    for fold, val_df in enumerate(vals, start=1):
        print("=" * 30)
        print(f"Fold {fold} Validation")
        report_nodule_characteristics(val_df)

if __name__ == "__main__":
    main()