from evaluate.args_parser import get_kfold_evaluate_args
from evaluate.evaluation_pipeline import run_validate_test_kfold_validation

def main():
    args = get_kfold_evaluate_args()
    run_validate_test_kfold_validation(**vars(args))
    
if __name__ == "__main__":
    main()