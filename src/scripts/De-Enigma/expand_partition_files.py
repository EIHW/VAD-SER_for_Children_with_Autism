from os.path import basename, splitext

import numpy as np

from src.utils.path_utils import get_system_dependendent_paths, make_directory
from src.utils.constants import *
import pandas as pd
from glob import glob

nas_dir, code_dir = get_system_dependendent_paths(LOCAL)
audio_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/audio_normalised/"
partitioning_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/data_split_files/"

partitions = ["train", "devel", "test"]

for partition in partitions:
    partition_sessions = []
    partition_df = pd.read_csv(partitioning_dir + partition + ".csv", header=None)
    partition_children = partition_df.iloc[:].values
    partition_children = np.squeeze(partition_children)
    for child in partition_children:
        x = audio_dir + child + "*"
        files = glob(audio_dir + child + "*_4*")
        for file in files:
            child_session_id = basename(file)[:8]
            partition_sessions.append(child_session_id)
    session_df = pd.DataFrame(partition_sessions)
    out_file = partitioning_dir + partition + "_sessions" + ".csv"
    session_df.to_csv(out_file, index=False, header=False)
