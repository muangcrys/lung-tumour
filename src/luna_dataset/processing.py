import pandas as pd
from pathlib import Path
from utility.paths import PathList

def patients_stratifying_key(gender:str, label:int):
    return gender + "_" + str(label)

def add_patients_stratifying_key(dataframe: pd.DataFrame,):
    df = dataframe.copy()
    df["stratifying_key"] = df.apply(lambda row: patients_stratifying_key(row["Gender"], row["label"]), axis=1)
    return df


def process_patients(output_csv: str|Path = None, annotation_csv: str|Path = None, write_csv = True):
    relevant_columns = ["PatientID", "Gender", "label"]
    if output_csv is None:
        output_csv = PathList.patients_csv
    if annotation_csv is None:
        annotation_csv = PathList.raw_luna_annotation_csv

    annotation_csv: Path = Path(annotation_csv)
    output_csv: Path = Path(output_csv)

    annotation_pd = pd.read_csv(annotation_csv)
    patient_pd = annotation_pd[relevant_columns].copy().groupby("PatientID").first().reset_index()

    output = add_patients_stratifying_key(patient_pd)

    if write_csv:
        output.to_csv(output_csv, index=False)

    return output






