from utility.k_fold import get_results_from_kfold
import argparse

def get_extract_kfold_results_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold_directory', type=str, required=False, default=None,)
    parser.add_argument('--save_target', type=str, required=False, default=None,)
    parser.add_argument('--source', type=str, required=False, default="kfold", choices=["kfold", "kfold_2stage"], help="Source of the k-fold results. Can be 'kfold' or 'kfold_2stage'.")
    parser.add_argument('--prefix', type=str, required=False, default=None,)
    return parser

def main():
    parser = get_extract_kfold_results_parser()
    args = parser.parse_args()
    get_results_from_kfold(**vars(args))

if __name__ == "__main__":
    main()