from utility.paths import PathList
from pathlib import Path
from typing import Union
import numpy as np


def resolve_file_name(nodule_id: str):
    extension = ".npy"
    return nodule_id + extension

def pull_data_raw(nodule_id: str, image_dir: str|Path = None):
    if image_dir is None:
        image_dir = PathList.raw_luna_image_dir

    image_dir: Path = Path(image_dir)

    target_file = image_dir / resolve_file_name(nodule_id)
    npy = np.load(target_file)
    return npy

