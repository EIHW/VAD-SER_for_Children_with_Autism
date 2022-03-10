from os.path import basename, splitext
from glob import glob
import pandas as pd
from src.utils.csv_utils import get_delimeter
import numpy as np

def parse_deenigma_file_info_from_path(path):
    info_list = splitext(basename(path))[0].split("_")
    info_dict = {}
    info_dict["child"] = info_list[0]
    info_dict["session"] = info_list[1]
    if len(info_list) > 2:
        info = info_list[2]
        info = splitext(info)[0]
        if len(info) == 1 and info.isnumeric() and int(info) <=4:
            info_dict["microphone"] = info
        elif len(info) > 1 and info.isnumeric():
            info_dict["chunk"] = info
    if len(info_list) > 3:
        info = info_list[3]
        info = splitext(info)[0]
        if len(info) > 1 and info.isnumeric():
            info_dict["chunk"] = info
    return info_dict

def convert_deenigma_chunked_feature_path_to_label_path(feature_path, label_dir, label_extension=".csv"):
    info_dict = parse_deenigma_file_info_from_path(feature_path)
    if "chunk" in info_dict.keys():
        label_path = label_dir + "_".join([info_dict["child"], info_dict["session"], info_dict["chunk"]]) + label_extension
    else:
        label_path = label_dir + "_".join(
            [info_dict["child"], info_dict["session"]]) + label_extension
    return label_path

"""def convert_deenigma_feature_path_to_label_path(feature_path, label_dir, label_extension=".csv"):
    info_dict = parse_deenigma_file_info_from_path(feature_path)
    label_path = label_dir + "_".join([info_dict["child"], info_dict["session"]]) + label_extension
    return label_path
"""

def convert_feature_path_to_label_path(feature_path, label_dir, label_extension=".csv"):
    label_path = splitext(splitext(basename(feature_path))[0])[0] + label_extension
    # this one
    label_path = label_dir + label_path.replace("_", " ", 1)
    return label_path

def add_original_labels_to_csv_file(evaluation_dir, original_label_dir, test_data_path):
    """
    This function is implemented for the VAD task and adds acolumn with the original majortiy label
    :param evaluation_dir: directory with the (second-wise) csv files
    :param original_label_dir: directory with the second-chunks and correct labels.
    :return: None, saves the updated files
    """
    df = pd.read_csv(test_data_path)
    session = df.iloc[0,0]
    original_csv_files = glob(original_label_dir + session + "*.csv")
    original_csv_files.sort()
    original_labels = []
    original_label_time = []
    all_labels = [[], [], [], []]
    all_label_times = [[], [], [], []]
    for original_csv_file in original_csv_files:
        original_label_df = pd.read_csv(original_csv_file)
        labels = original_label_df["label"].values
        label_time_dict = {}
        for label in labels:
            if not label in label_time_dict.keys():
                label_time_dict[label] = 0.
            label_time_dict[label] += 0.01
        max_time = 0.
        for label in label_time_dict.keys():
            if label_time_dict[label] >= max_time:
                max_label = label
                max_time = label_time_dict[label]
        original_labels.append(max_label)
        original_label_time.append(max_time)
        for i, label in enumerate(label_time_dict.keys()):
            all_labels[i].append(label)
            all_label_times[i].append(label_time_dict[label])
        while i < len(all_labels) - 1:
            i += 1
            all_labels[i].append("None")
            all_label_times[i].append(0)


    original_labels = np.array(original_labels)
    original_label_time = np.array(original_label_time)
    threshold_files = glob(evaluation_dir + "*.csv")
    threshold_files.sort()
    for threshold_file in threshold_files:
        threshold_df = pd.read_csv(threshold_file)

        threshold_df["original_label"] = original_labels
        threshold_df["original_label_time"] = original_label_time
        for i in range(len(all_labels)):
            x = all_labels[i]
            threshold_df["label_{}".format(i)] = np.array(all_labels[i])
            threshold_df["label_time_{}".format(i)] = np.array(all_label_times[i])
        threshold_df.to_csv(threshold_file, index=False)


def get_partition_files(file_dir, file_ext, partition_file):
    """
    Determines list of files for given partition as specified by partition_file
    :param file_dir: directory with files
    :param partition_file: partition info (usually allowed file prefixes)
    :return: list of partition files
    """
    data_frame = pd.read_csv(partition_file, delimiter=get_delimeter(partition_file), header=None)
    # This part is quite specific to de-enigma organisation.
    data_identifications = data_frame.iloc[:, 0].values


    out_files = []
    files = glob(file_dir + "*" + file_ext)
    if len(data_identifications) == 0:
        return files

    partition_identification_len = len(data_identifications[0])
    for file in files:
        file_identifier = basename(file)[:partition_identification_len]
        if file_identifier in data_identifications:
            out_files.append(file)

    out_files.sort()
    return out_files

def combine_interpolated_valence_arousal_files(arousal_raw_dir, valence_raw_dir, label_out_dir, interval):
    max_arousal_value = 0
    min_arousal_value = 0
    max_valence_value = 0
    min_valence_value = 0
    arousal_files = glob(arousal_raw_dir + "*.csv")
    arousal_files.sort()
    valence_files = glob(valence_raw_dir + "*.csv")
    valence_files.sort()
    for arousal_file, valence_file in zip(arousal_files, valence_files):
        arousal_df_raw = pd.read_csv(arousal_file)
        valence_df_raw = pd.read_csv(valence_file)
        max_arousal_value = max(np.max(arousal_df_raw.values[:,-1]), max_arousal_value) # s
        max_valence_value = max(np.max(valence_df_raw.values[:,-1]), max_valence_value)
        min_arousal_value = min(np.min(arousal_df_raw.values[:, -1]), min_arousal_value)  # s
        min_valence_value = max(np.min(valence_df_raw.values[:, -1]), min_valence_value)
    arousal_normalise = max(abs(min_arousal_value), max_arousal_value)
    valence_normalise = max(abs(min_valence_value), max_valence_value)
    for arousal_file, valence_file in zip(arousal_files, valence_files):
        # assert labels are available for both files
        if basename(arousal_file) != basename(valence_file):
            print(basename(arousal_file) + " is not equal to " + basename(valence_file))
        print("Doing " + basename(arousal_file))
        arousal_df_raw = pd.read_csv(arousal_file)
        valence_df_raw = pd.read_csv(valence_file)
        arousal_file_len = arousal_df_raw.values[-1,0]# s
        valence_file_len = valence_df_raw.values[-1,0]
        # assert same length of valence and arousal annotations
        if valence_file_len != arousal_file_len:
            print(str(valence_file_len) + " is not equally long as " + str(arousal_file_len))
        num_labels_out = int(np.ceil(valence_file_len/interval))


        # calculate averages within interval size
        label_data_out = np.zeros((num_labels_out, 3))

        if arousal_df_raw.shape[0] > num_labels_out:
            label_out_step = 0
            arousal_out = [arousal_df_raw.iloc[0, 0]]
            valence_out = [valence_df_raw.iloc[0, 0]]
            for i in range(arousal_df_raw.shape[0]):
                current_time = arousal_df_raw.iloc[i, 0]
                if current_time < (label_out_step + 1) * interval:
                    arousal_out.append(arousal_df_raw.iloc[i,1])
                    valence_out.append(valence_df_raw.iloc[i, 1])
                else:
                    label_data_out[label_out_step] = np.array([label_out_step * interval, np.mean(arousal_out)/arousal_normalise, np.mean(valence_out)/valence_normalise])
                    arousal_out = [arousal_df_raw.iloc[i, 1]]
                    valence_out = [valence_df_raw.iloc[i, 1]]
                    label_out_step += 1
            if label_out_step < num_labels_out:
                label_data_out[label_out_step] = np.array(
                    [label_out_step * interval, np.mean(arousal_out) / arousal_normalise,
                     np.mean(valence_out) / valence_normalise])
        else:
            label_raw_step = 0
            for label_out_step in range(num_labels_out):
                current_time = arousal_df_raw.iloc[label_raw_step, 0]
                if label_out_step * interval > current_time:
                    label_raw_step += 1
                arousal_value = arousal_df_raw.iloc[label_raw_step,1]
                valence_value = valence_df_raw.iloc[label_raw_step, 1]
                label_data_out[label_out_step] = np.array(
                    [label_out_step * interval, arousal_value / arousal_normalise,
                     valence_value / valence_normalise])

        label_out_df = pd.DataFrame(label_data_out, columns=["time", "arousal", "valence"])
        label_out_path = label_out_dir + basename(arousal_file)[:8] + ".csv"
        label_out_df.to_csv(label_out_path, index=False)

def convert_speech_detection_labels(label_raw_dir, label_out_dir, interval, speaker):
    label_raw_files = glob(label_raw_dir + "*")
    label_raw_files.sort()
    for label_raw_file in label_raw_files:
        print("Doing " + label_raw_file)
        label_df = pd.read_csv(label_raw_file, delimiter=get_delimeter(label_raw_file))
        total_time = label_df.iloc[-1, 1]
        out_labels = np.zeros((int(np.ceil(total_time/interval)), 2))
        current_annotation_idx = 0
        current_start = -1
        current_end = -1

        for i in np.arange(0, out_labels.shape[0]):
            current_time = i * interval
            if current_time > current_end:
                current_start = label_df.iloc[current_annotation_idx, 0]
                current_end = label_df.iloc[current_annotation_idx, 1]
                current_label = label_df.iloc[current_annotation_idx, 2]
                current_annotation_idx += 1
            out_labels[i, 0] = current_time
            if current_time > current_start and current_label in speaker:
                out_labels[i, 1] = 1
            else:
                out_labels[i, 1] = 0
        out_df = pd.DataFrame(out_labels, columns=["time", "vocalisation"])
        outfile = label_out_dir + basename(label_raw_file)
        out_df.to_csv(outfile, index=False)
