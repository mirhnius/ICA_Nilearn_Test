import numpy as np
import pandas as pd

subject_df = pd.read_csv("/data/origami/niusha/input/subject_IDs.csv")

N = len(subject_df.ID)
ID_map = dict(zip(range(N),subject_df.ID))

Healthy_index = np.ravel(np.where(subject_df.PD == 0))
Healthy_subject = subject_df.ID.iloc[Healthy_index]

PD_index =  np.ravel(np.where(subject_df.PD == 1))
PD_subject = subject_df.ID.iloc[PD_index]


