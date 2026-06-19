from functools import wraps


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

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name.startswith("_") or name in {"add_prefix", "__class__"}:
            return attr

        if callable(attr):
            @wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)

                if isinstance(result, str):
                    return ClassifierOnlyFileNameResolver.add_prefix(result)

                return result

            return wrapper

        return attr
