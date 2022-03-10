from src.utils.constants import *
from src.utils.path_utils import get_system_dependendent_paths, make_directory
from glob import glob
from src.utils.csv_utils import get_delimeter
import pandas as pd
from os.path import basename

executing_pc = LOCAL
label = "arousal"


nas_dir, code_dir = get_system_dependendent_paths(executing_pc)
data_root_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/"
feature_dir = data_root_dir + "chunked_labels_1.0_features/"
label_dir = data_root_dir + "chunked_labels_1.0_labels/"
out_feature_file = data_root_dir + "chunked_labels_1.0_features_all.csv"
out_label_file = data_root_dir + "chunked_labels_1.0_labels_all.csv"

feature_files = glob(feature_dir + "*")
feature_files.sort()

label_files = glob(label_dir + "*")
label_files.sort()

# feature_df = pd.DataFrame()
# delim = get_delimeter(feature_files[0])
# for feature_file in feature_files:
#     print(basename(feature_file))
#     if feature_df.shape[0] == 0:
#         feature_df = pd.read_csv(feature_file, delimiter=delim)
#     else:
#         feature_df = feature_df.append(pd.read_csv(feature_file, delimiter=delim))
# feature_df.to_csv(out_feature_file, index=False)


label_df = pd.DataFrame()
delim = get_delimeter(label_files[0])
for label_file in label_files:
    print(basename(label_file))
    if label_df.shape[0] == 0:
        label_df = pd.read_csv(label_file, delimiter=delim)
    else:
        label_df = label_df.append(pd.read_csv(label_file, delimiter=delim))
label_df.to_csv(out_label_file, index=False)

