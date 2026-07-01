from pathlib import Path
from utility.paths import PathList
import json
import pandas as pd

def flatten_results(result_dict):
    flattened = {}
    for key, value in result_dict.items():
        if key == "confusion_matrix":
            flattened["TN"] = value["TN"]
            flattened["FP"] = value["FP"]
            flattened["FN"] = value["FN"]
            flattened["TP"] = value["TP"]
        else:
            flattened[key] = value

    return flattened


def get_results_from_kfold(kfold_directory: Path = None,
                           save_file: bool = True,
                           save_target: str | Path = None,):
    if kfold_directory is None:
        kfold_directory = PathList.saved_kfold_weights_dir
    else:
        kfold_directory = Path(kfold_directory)

    dir_glob = kfold_directory.glob("*")
    dirs = sorted([p for p in dir_glob if p.is_dir()], key=lambda p: p.stem)  # sort by model name
    rows = []

    for model in dirs:
        model_name = model.stem
        print("=" * 40)
        print(f"Processing {model_name}")
        # find latest fold directory
        latest_dir = max(
            (p for p in model.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
        print(f"Using latest fold directory: {latest_dir}")
        
        folds_glob = latest_dir.glob("fold_*")
        folds = sorted([p for p in folds_glob if p.is_dir()], key=lambda p: int(p.stem.split("_")[-1]))
        for fold in folds:
            fold_num = fold.stem.split("_")[-1]

            print(f"Processing fold {fold}")

            # get validation metric and test metric
            validation_metric_glob = [
                p for p in fold.glob("#BEST_*_validate_metrics.json")
                if p.is_file()
            ]
            test_metric_glob = [
                p for p in fold.glob("#BEST_*_test_metrics.json")
                if p.is_file()
            ]

            if len(validation_metric_glob) > 1:
                print(f"Warning: More than one validation metric file found for model {model_name} fold {fold_num}. Using the first one.")
                print(validation_metric_glob)
                validation_metric_file = validation_metric_glob[0]
            elif len(validation_metric_glob) == 0:
                print(f"Warning: No validation metric file found for model {model_name} fold {fold_num}.")
                raise FileNotFoundError
            else:
                validation_metric_file = validation_metric_glob[0]

            if len(test_metric_glob) > 1:
                print(f"Warning: More than one test metric file found for model {model_name} fold {fold_num}.")
                print(test_metric_glob)
                test_metric_file = test_metric_glob[0]
            elif len(test_metric_glob) == 0:
                print(f"Warning: No test metric file found for model {model_name} fold {fold_num}.")
                raise FileNotFoundError
            else:
                test_metric_file = test_metric_glob[0]

            # get best epoch
            validate_best_epoch = int(validation_metric_file.stem.split("_")[1])
            test_best_epoch = int(test_metric_file.stem.split("_")[1])
            if validate_best_epoch != test_best_epoch:
                print(f"Mismatch between validation ({validate_best_epoch}) and test ({test_best_epoch}) epochs")
            best_epoch = test_best_epoch

            # read the metrics
            with validation_metric_file.open("r") as f:
                validation_metric = json.load(f)

            with test_metric_file.open("r") as f:
                test_metric = json.load(f)

            # flatten metrics
            validation_flattened = flatten_results(validation_metric)
            test_flattened = flatten_results(test_metric)

            # construct row
            row = {
                "model_name": model_name,
                "fold": fold_num,
                "best_epoch": best_epoch,
                **{f"validate_{k}": v for k, v in validation_flattened.items()},
                **{f"test_{k}": v for k, v in test_flattened.items()},
            }
            rows.append(row)

    # create dataframe
    df = pd.DataFrame(rows)

    if save_file:
        save_target = Path(save_target) if save_target else PathList.k_fold_output_dir

    save_target.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_target, index=False)

    return df


def main():
    get_results_from_kfold()

if __name__ == "__main__":
    main()

