import argparse
from luna_dataset.processing import process_patients

def main():
    parser = argparse.ArgumentParser(description='Process patients data from raw annotation')

    parser.add_argument('--input', type=str, required=False, help='input csv file')
    parser.add_argument('--output', type=str, required=False, help='output csv file')
    args = parser.parse_args()

    input_patients_csv = args.input
    output_patients_csv = args.output

    process_patients(input_patients_csv, output_patients_csv)

if __name__ == "__main__":
    main()

