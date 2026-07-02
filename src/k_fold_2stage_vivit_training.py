from training_k_fold.training_2stages_k_fold import *
from training_k_fold.args_parser import get_kf_vivit_2stage_args

def main():
    args = get_kf_vivit_2stage_args()
    _ = k_fold_2_stage_vivit_wrapper(**vars(args))

if __name__ == "__main__":
    main()