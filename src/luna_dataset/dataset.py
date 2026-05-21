import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from pathlib import Path
from typing import cast, Literal, List
from utility.paths import PathList


class LunaColumns:
    filename = "AnnotationID"
    label = "label"


class LunaDataset(Dataset):
    def __init__(self,
                 annotation_file: str | Path = None,
                 image_dir: str | Path = None,
                 transform=None,
                 split: Literal["train", "validate", "test"] = None,
                 seed: int = 4242):

        if annotation_file is None:
            if split is None:
                raise ValueError("Either annotation_file or split must be provided.")
            seed_dir = "SEED_" + str(seed)
            annotation_file_name = split + "_annotations.csv"
            self.annotation_file = PathList.processed_luna_data_dir / seed_dir / annotation_file_name
        else:
            self.annotation_file = Path(annotation_file)

        if not Path.is_file(self.annotation_file):
            raise FileNotFoundError(f"File {self.annotation_file} not found.")

        if image_dir is None:
            self.image_dir = PathList.raw_luna_image_dir
        else:
            self.image_dir = Path(image_dir)
        self.transform = transform

        # load annotation df
        self.annotations: DataFrame = cast(DataFrame, pd.read_csv(self.annotation_file))

        # check that all files are there
        missing_files = self.get_missing_files()
        if len(missing_files) > 0:
            raise FileNotFoundError(f"Missing files: {missing_files}")

    def get_missing_files(self) -> List[str]:
        # check that all image in annotation df has a file in image dir
        annotation_ids = self.annotations[LunaColumns.filename]

        # map
        target_files = annotation_ids.map(lambda annotation_id: self.get_image_path(annotation_id))
        missing_files = target_files.filter(lambda file: not Path(file).exists())

        return missing_files.to_list()

    def __len__(self):
        return len(self.annotations)

    def get_image_path_from_row(self, row: pd.Series):
        annotation_id = row[LunaColumns.filename]
        return self.get_image_path(annotation_id)

    def get_image_path(self, annotation_id: str):
        image_file_name = annotation_id + ".npy"
        image_path = self.image_dir / image_file_name
        return image_path

    def get_label(self, row: pd.Series):
        label = row[LunaColumns.label]
        return int(label)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = self.get_image_path_from_row(row)
        label = self.get_label(row)

        # read the file
        image = np.load(image_path)

        # add channel dimension
        image = image[None, :]
        # transform image if needed
        if self.transform:
            image = self.transform(image)

        return image, label

