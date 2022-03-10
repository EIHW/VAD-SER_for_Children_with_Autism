from glob import glob

import numpy as np
import pandas as pd
from os.path import basename, splitext
from src.utils.csv_utils import get_delimeter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def average_feature_dir(in_dir, out_dir, pool_size, start_column = 0, end_column=-1, file_ext=".csv", header=True):
    feature_files = glob(in_dir + "*" + file_ext)
    feature_files.sort()
    out_df_all_features = pd.DataFrame()
    file_info = []
    j = 0
    for file in feature_files:
        print(file)
        if header:
            feature_df = pd.read_csv(file, delimiter=get_delimeter(file))
        else:
            feature_df = pd.read_csv(file, delimiter=get_delimeter(file), header=None)
        if len(out_df_all_features) == 0:
            #columns = np.concatenate((feature_df.columns.values, ["id"]))
            columns = feature_df.columns.values
            out_df_all_features = pd.DataFrame(columns=columns)
        feature_data_in = feature_df.iloc[:, start_column:end_column].values
        out_lines = int(np.ceil(feature_df.shape[0] / pool_size))
        #feature_data_out = np.zeros(out_lines, feature_df.shape[1])
        feature_df_out = feature_df.iloc[:out_lines, :]
        # WTF is going on here?
        file_info.append((out_lines, out_dir + basename(file)))
        if pool_size > 1:
            for i in range(out_lines):
                #feature_df[i,0] = feature_df.iloc[0,0].values
                feature_df_out.iloc[i, start_column:end_column] = np.mean(feature_data_in[i * pool_size:(i + 1) * pool_size], axis=0)
        out_df_all_features = out_df_all_features.append(feature_df_out, ignore_index=True)
        #
        # if j >= 4:
        #     break
        # j += 1


    # TODO: Feature scaler needs to be based only on training data
    all_features = out_df_all_features.iloc[:, start_column:end_column].values

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(all_features)
    all_features_scaled = feature_scaler.transform(all_features)
    out_df_all_features.iloc[:, start_column:end_column] = all_features_scaled

    line_idx = 0
    for n_lines, out_file_path in file_info:
        out_df =out_df_all_features.iloc[line_idx:n_lines+line_idx]
        out_df.to_csv(out_file_path, index=False)
        line_idx += n_lines



