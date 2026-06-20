class FileNameResolver:
    @staticmethod
    def get_checkpoint_name(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_stats_filename(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return epoch_str + "_" + "stats.csv"

    @staticmethod
    def get_best_checkpoint_name(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return "#BEST_" + epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_best_json_filename(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return "#BEST_" + epoch_str + "_" + "details.json"

    @staticmethod
    def get_final_checkpoint_name(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "ckt.pth"

    @staticmethod
    def get_final_json_filename(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "details.json"

    @staticmethod
    def get_final_stats_filename(epoch: int | str) -> str:
        epoch_str = str(epoch)
        return "#FINAL_" + epoch_str + "_" + "stats.csv"

    @staticmethod
    def get_training_configs_filename() -> str:
        return "###TrainingConfigs.json"

    @staticmethod
    def get_2stage_training_configs_filename() -> str:
        return "###TrainingConfigs.json"


class ClassifierOnlyFileNameResolver(FileNameResolver):
    @staticmethod
    def add_prefix(filename: str) -> str:
        return "~ClassifierOnly~" + filename

    @staticmethod
    def get_checkpoint_name(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_checkpoint_name(epoch)
        )

    @staticmethod
    def get_stats_filename(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_stats_filename(epoch)
        )

    @staticmethod
    def get_best_checkpoint_name(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_best_checkpoint_name(epoch)
        )

    @staticmethod
    def get_best_json_filename(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_best_json_filename(epoch)
        )

    @staticmethod
    def get_final_checkpoint_name(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_final_checkpoint_name(epoch)
        )

    @staticmethod
    def get_final_json_filename(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_final_json_filename(epoch)
        )

    @staticmethod
    def get_final_stats_filename(epoch: int | str) -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_final_stats_filename(epoch)
        )

    @staticmethod
    def get_training_configs_filename() -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_training_configs_filename()
        )

    @staticmethod
    def get_2stage_training_configs_filename() -> str:
        return ClassifierOnlyFileNameResolver.add_prefix(
            FileNameResolver.get_2stage_training_configs_filename()
        )
