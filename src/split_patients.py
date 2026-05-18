import argparse
from luna_dataset.splitting import split_patients, split_train_patients
from utility.paths import PathList

def report_characteristics(df):
    patients = df.shape[0]
    males = df[df["Gender"] == "Male"].shape[0]
    females = df[df["Gender"] == "Female"].shape[0]
    positive = df[df["label"] == 1].shape[0]
    negative = df[df["label"] == 0].shape[0]

    print(f"Total Patients: {patients}")
    print(f"Males: {males}, Females: {females} --> Ratio: {males/patients}M to {females/patients}F")
    print(f"Positive: {positive}, Negative: {negative} --> Ratio: {positive/negative}")

def main():
    parser = argparse.ArgumentParser(description='Split patients into train and test')

    parser.add_argument('--input', type=str, help='input patients csv file', required=False,
                        default=PathList.patients_csv)
    parser.add_argument('--stratify_by', type=str, help='column in the csv file to stratify by', required=False,
                        default="stratifying_key")
    parser.add_argument('--output_dir', type=str, help='output directory for train and test csv files', required=False)
    parser.add_argument('--seed', type=int, help='random seed for splitting', required=False, default=4242)
    parser.add_argument('--test_size', type=float, help='percentage of patients to use for testing', required=False,
                        default=0.2)
    parser.add_argument('--validate_size', type=float, help='percentage of dev patients to use for validation', required=False,
                        default=0.25)

    args = parser.parse_args()
    dev, test = split_patients(patients_csv=args.input,
                                 output_dir=args.output_dir,
                                 stratify_by=args.stratify_by,
                                 seed=args.seed,
                                 test_size=args.test_size,
                                 save_file=True)

    # do some reports
    print("=" * 30)
    print("Development Set Characteristics (Patients):")
    report_characteristics(dev)
    print("=" * 30)
    print("Test Set Characteristics (Patients):")
    report_characteristics(test)


    # now split dev set into train and validation
    print("=" * 30)
    print("Splitting Development Set into Train and Validation...")
    train, validate = split_train_patients(patients_train_csv=dev,
                                           output_dir=args.output_dir,
                                           stratify_by=args.stratify_by,
                                           seed=args.seed,
                                           validate_size=args.validate_size,
                                           save_file=True)

    # report again
    print("=" * 30)
    print("Split Train Set Characteristics (Patients):")
    report_characteristics(train)
    print("=" * 30)
    print("Split Validation Set Characteristics (Patients):")
    report_characteristics(validate)


if __name__ == "__main__":
    main()
