class FileNameResolver:
    @staticmethod
    def get_checkpoint_name(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_stats_filename(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return epoch_str + "_" + "stats.csv"

    @staticmethod
    def get_best_checkpoint_name(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return "#BEST_" + epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_best_json_filename(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return "#BEST_" + epoch_str + "_" + "details.json"

    @staticmethod
    def get_final_checkpoint_name(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_final_json_filename(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "details.json"

    @staticmethod
    def get_final_stats_filename(epoch: int|str) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "stats.csv"

    @staticmethod
    def get_training_configs_filename() -> str:
        return "###TrainingConfigs.json"
