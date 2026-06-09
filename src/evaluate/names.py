class ColumnNames:
    labels = "labels"
    predictions = "predictions"


class MetricFiles:
    @staticmethod
    def get_final_loss_curve_filename(final_epoch: int | str) -> str:
        final_epoch_str = str(final_epoch)
        return "#FINAL_" + final_epoch_str + "_" + "loss_curve.pdf"

    ################################## BEST ##################################

    @staticmethod
    def get_best_prediction_filename(best_epoch: int | str) -> str:
        best_epoch_str = str(best_epoch)
        return "#BEST_" + best_epoch_str + "_" + "predictions.csv"

    @staticmethod
    def get_best_metrics_filename(best_epoch: int | str) -> str:
        best_epoch_str = str(best_epoch)
        return "#BEST_" + best_epoch_str + "_" + "metrics.json"

    @staticmethod
    def get_best_roc_filename(best_epoch: int | str) -> str:
        best_epoch_str = str(best_epoch)
        return "#BEST_" + best_epoch_str + "_" + "roc.pdf"

    @staticmethod
    def get_best_pr_filename(best_epoch: int | str) -> str:
        best_epoch_str = str(best_epoch)
        return "#BEST_" + best_epoch_str + "_" + "pr.pdf"

    @staticmethod
    def get_best_confmat_filename(best_epoch: int | str) -> str:
        best_epoch_str = str(best_epoch)
        return "#BEST_" + best_epoch_str + "_" + "confmat.pdf"

    ################################## FINAL ##################################

    @staticmethod
    def get_final_prediction_filename(final_epoch: int | str) -> str:
        final_epoch_str = str(final_epoch)
        return "#FINAL_" + final_epoch_str + "_" + "predictions.csv"

    @staticmethod
    def get_final_metrics_filename(final_epoch: int | str) -> str:
        final_epoch_str = str(final_epoch)
        return "#FINAL_" + final_epoch_str + "_" + "metrics.json"

    @staticmethod
    def get_final_roc_filename(final_epoch: int | str) -> str:
        final_epoch_str = str(final_epoch)
        return "#FINAL_" + final_epoch_str + "_" + "roc.pdf"

    @staticmethod
    def get_final_pr_filename(final_epoch: int | str) -> str:
        final_epoch_str = str(final_epoch)
        return "#FINAL_" + final_epoch_str + "_" + "pr.pdf"

    @staticmethod
    def get_final_confmat_filename(final_epoch: int | str) -> str:
        final_epoch_str = str(final_epoch)
        return "#FINAL_" + final_epoch_str + "_" + "confmat.pdf"
