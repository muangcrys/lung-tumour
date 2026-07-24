from training_k_fold.training_luna16_k_fold import *
from training_k_fold.args_parser import get_kf_luna16_double_args

def main():
    args = get_kf_luna16_double_args()
    _ = k_fold_luna16_double_wrapper(**vars(args))

if __name__ == "__main__":
    main()