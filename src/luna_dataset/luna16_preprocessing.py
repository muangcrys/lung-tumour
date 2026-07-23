"""
    transform and itk_image_to_numpy_image from https://github.com/DIAGNijmegen/luna25-baseline-public/tree/main
"""

import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Dict
from pathlib import Path
from utility.paths import PathList


def transform(input_image, point):
    """
    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points
    Returns
    -------
    tNumpyOrigin
    """
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )


def itk_image_to_numpy_image(input_image: sitk.Image) -> Tuple[np.ndarray, Dict]:
    """
    Parameters
    ----------
    input_image: SimpleITK image
    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format
    """

    numpyImage = sitk.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header


def process_niigz_to_npy(niigz_directory: Path = PathList.luna16_niigz_dir,
                         npy_output_directory: Path = PathList.luna16_img_npy_dir,
                         metadata_output_directory: Path = PathList.luna16_metadata_npy_dir):
    niigz_directory = Path(niigz_directory)
    npy_output_directory = Path(npy_output_directory)
    metadata_output_directory = Path(metadata_output_directory)

    npy_output_directory.mkdir(parents=True, exist_ok=True)
    metadata_output_directory.mkdir(parents=True, exist_ok=True)
    img_paths = niigz_directory.glob("*.nii.gz")

    for img_path in img_paths:
        print(f"Processing {img_path}")
        # get the name
        img_name = img_path.name
        img_output_name = img_name.replace(".nii.gz", ".npy")
        img_output_path = Path(npy_output_directory / img_output_name)
        metadata_output_path = Path(metadata_output_directory / img_output_name)

        img_sitk = sitk.ReadImage(img_path)
        img, header = itk_image_to_numpy_image(img_sitk)

        np.save(img_output_path, img)
        np.save(metadata_output_path, header)
        print(f">>> saved image data to {img_output_path}")
        print(f">>> saved metadata data to {metadata_output_path}")

def malignancy_designation(score:float) -> int:
    threshold = 3
    if score < threshold:
        return 0
    else:
        return 1

def transform_npy_annotation_to_csv(annotation_npy: Path = PathList.luna16_raw_annotation_npy,
                                    output_npy: Path = PathList.luna16_annotation_csv):
    annotation_npy = Path(annotation_npy)
    output_npy = Path(output_npy)

    output_npy.parent.mkdir(parents=True, exist_ok=True)

    raw_annotation = np.load(annotation_npy, allow_pickle=True)
    annotation_df = pd.DataFrame()

    annotation_df["SeriesInstanceUID"] = [d["SeriesInstanceUID"] for d in raw_annotation]
    annotation_df["PatientID"] = [d["SeriesInstanceUID"] for d in raw_annotation]
    annotation_df["AnnotationID"] = [d["Filename"].replace(".nii.gz", "") for d in raw_annotation]
    annotation_df["AnnotationCount"] = [len(d["Malignancy"]) for d in raw_annotation]
    annotation_df["MeanMalignancyScore"] = [np.mean(d["Malignancy"]) for d in raw_annotation]
    annotation_df["RoundedMalignancyScore"] = [int(round(np.mean(d["Malignancy"]))) for d in raw_annotation]

    # exclude annotations with rounded score = 3
    annotation_df = annotation_df[annotation_df["RoundedMalignancyScore"] != 3].copy()
    annotation_df["label"] = annotation_df["RoundedMalignancyScore"].apply(malignancy_designation)

    # save to output
    annotation_df.to_csv(output_npy, index=False)


