from evaluate.args_parser import get_evaluate_args
from evaluate.evaluation_pipeline import run_evaluation_on_model_directory

def main():
    args = get_evaluate_args()
    run_evaluation_on_model_directory(**vars(args))

if __name__ == "__main__":
    main()