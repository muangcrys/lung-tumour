import tomllib
import pandas as pd
import numpy as np
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "windows.toml"

def resolve_data_path(data_path=None, config_path=None):
    if data_path is not None:
        return data_path

    if config_path is None:
        env_config_path = os.getenv("LUNA_CONFIG_PATH")
        if env_config_path is None:
            # default path
            print(f"env_config_path is not defined -> defaulting to {DEFAULT_CONFIG_PATH}")
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = Path(env_config_path)
    else:
        config_path = Path(config_path)


    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config["dataset"]["images"], config["dataset"]["annotations"]

def pull_data(nodule_id:str, data_path=None, config_path=None):
    if data_path is not None:
        return data_path