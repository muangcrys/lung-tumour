class ColumnNames:
    labels = "labels"
    predictions = "predictions"


class MetricFiles:

    ################################## BEST ##################################

    @staticmethod
    def get_best_prediction_filename(best_epoch: int | str,
                                     split_name: str = None) -> str:
        best_epoch_str = str(best_epoch)
        if split_name is None or split_name == "":
            return "#BEST_" + best_epoch_str + "_" + "predictions.csv"
        else:
            return "#BEST_" + best_epoch_str + "_" + split_name + "_predictions.csv"

    @staticmethod
    def get_best_metrics_filename(best_epoch: int | str,
                                  split_name: str = None) -> str:
        best_epoch_str = str(best_epoch)
        if split_name is None or split_name == "":
            return "#BEST_" + best_epoch_str + "_" + "metrics.json"
        else:
            return "#BEST_" + best_epoch_str + "_" + split_name + "_metrics.json"

    @staticmethod
    def get_best_roc_filename(best_epoch: int | str,
                              split_name: str = None) -> str:
        best_epoch_str = str(best_epoch)
        if split_name is None or split_name == "":
            return "#BEST_" + best_epoch_str + "_" + "roc.pdf"
        else:
            return "#BEST_" + best_epoch_str + "_" + split_name + "_roc.pdf"

    @staticmethod
    def get_best_pr_filename(best_epoch: int | str,
                             split_name: str = None) -> str:
        best_epoch_str = str(best_epoch)
        if split_name is None or split_name == "":
            return "#BEST_" + best_epoch_str + "_" + "pr.pdf"
        else:
            return "#BEST_" + best_epoch_str + "_" + split_name + "_pr.pdf"

    @staticmethod
    def get_best_confmat_filename(best_epoch: int | str,
                                  split_name: str = None) -> str:
        best_epoch_str = str(best_epoch)
        if split_name is None or split_name == "":
            return "#BEST_" + best_epoch_str + "_" + "confmat.pdf"
        else:
            return "#BEST_" + best_epoch_str + "_" + split_name + "_confmat.pdf"

    ################################## FINAL ##################################

    @staticmethod
    def get_final_prediction_filename(final_epoch: int | str,
                                      split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "predictions.csv"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_predictions.csv"

    @staticmethod
    def get_final_metrics_filename(final_epoch: int | str,
                                   split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "metrics.json"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_metrics.json"

    @staticmethod
    def get_final_roc_filename(final_epoch: int | str,
                               split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "roc.pdf"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_roc.pdf"

    @staticmethod
    def get_final_pr_filename(final_epoch: int | str,
                              split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "pr.pdf"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_pr.pdf"

    @staticmethod
    def get_final_confmat_filename(final_epoch: int | str,
                                   split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "confmat.pdf"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_confmat.pdf"

    @staticmethod
    def get_final_loss_curve_filename(final_epoch: int | str,
                                      split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "loss_curve.pdf"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_loss_curve.pdf"

    @staticmethod
    def get_final_metrics_plot_filename(final_epoch: int | str,
                                        split_name: str = None) -> str:
        final_epoch_str = str(final_epoch)
        if split_name is None or split_name == "":
            return "#FINAL_" + final_epoch_str + "_" + "metrics.pdf"
        else:
            return "#FINAL_" + final_epoch_str + "_" + split_name + "_metrics.pdf"
