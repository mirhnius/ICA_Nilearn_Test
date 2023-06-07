import numpy as np
import pandas as pd
from nilearn.image import load_img

subject_df = pd.read_csv("/data/origami/niusha/input/subject_IDs.csv")
Original_DBMs = load_img("/data/origami/niusha/input/DBM_data.nii")

N = len(subject_df.ID)
ID_map = dict(zip(range(N),subject_df.ID))

Healthy_index = np.ravel(np.where(subject_df.PD == 0))
Healthy_subject = subject_df.ID.iloc[Healthy_index]

PD_index =  np.ravel(np.where(subject_df.PD == 1))
PD_subject = subject_df.ID.iloc[PD_index]


if __name__ == "__main__":
    from ica_module import io_path
    
    whole_original_path = io_path / "whole_original_data"
    whole_original_path_ICAs = whole_original_path / "ICAs"
    whole_original_path_PD = whole_original_path / "PD"
    whole_original_path_Healthy = whole_original_path / "Healty"

    if not whole_original_path.exists():
        whole_original_path.mkdir(parents=True, exist_ok=True)
        whole_original_path_ICAs.mkdir(parents=True, exist_ok=True)
        whole_original_path_PD.mkdir(parents=True, exist_ok=True)
        whole_original_path_Healthy.mkdir(parents=True, exist_ok=True)