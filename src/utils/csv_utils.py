import numpy as np
import pandas as pd
from os.path import basename, splitext



def get_delimeter(csv_file, num_lines=10):
    """
    TODO: Improve implementation
    Determine the delimiter of a csv_file. Read num_lines lines and search for occuring delimiter
    :param csv_file: path to csv file
    :param num_lines: number of lines to consider
    :return:
    """
    # order of potential delimiters according to my estimate of how likely they would appear without actually being the delimiter
    potential_delims = [";", "\t", ",", " "]
    lines = []
    with open(csv_file) as f:
        # read num_lines lines
        try:
            for _ in range(num_lines):
                line = f.readline()
                if len(line) > 2:
                    lines.append(line)
        except:
            pass
    if len(lines) == 0:
        print("Length of csv file is zero, can't extract delimeter: " + csv_file)
        return

    for delim in potential_delims:
        if check_potential_delimiter(lines, delim):
            return delim
    print("No appropriate delim was found for: " + csv_file)


def check_potential_delimiter(lines, delim):
    """
    TODO: Improve implementation
    Check whether delim is the delimiter for lines of a csv file. The delimiter should: 
    - Appear in every line at least once
    - Appear the same number of times in each line (?)
    :param lines: lines to check: list of strings
    :param delim: delimiter to check: string of length 1
    :return: True if delimiter is a possible delimiter, False otherwise
    """
    delim_counts = np.empty(len(lines))
    for i, line in enumerate(lines):
        delim_counts[i] = line.count(delim)
    if delim_counts[0] > 0 and (delim_counts == delim_counts[0]).all():
        return True
    else:
        return False

def csv_file_contains_string(csv_file, string):
    # returns True if string is contained in csv_file, else False
    with open(csv_file) as f:
        content = f.read()
    if string in content:
        return True
    else:
        return False

def count_label_in_csv_file(csv_file, label):
    """
    Counts the occurences of labels in  csv-file
    :param csv_file:
    :param label:
    :return:
    """
    data_frame = pd.read_csv(csv_file, delimiter=get_delimeter(csv_file))
    maximum_length_dataframe = 0
    # iterate over columns, cause we don't know in which column the labels are.
    # We thereby assume that the label is not in any different column
    for column in data_frame:
        data_frame_label_subset = data_frame[data_frame[column] == label]
        maximum_length_dataframe = max(maximum_length_dataframe, data_frame_label_subset.shape[0])
    # print(basename(csv_file)[:8] + ", " + label +  ": " + str(maximum_length_dataframe))
    return maximum_length_dataframe

def count_label_time_in_csv_file(csv_file, label):
    """
    Counts the occurences of labels in  csv-file
    :param csv_file:
    :param label:
    :return:
    """
    data_frame = pd.read_csv(csv_file, delimiter=get_delimeter(csv_file))
    maximum_length_dataframe = 0

    numerics = []

    # iterate over columns, cause we don't know in which column the labels are and what time and end columns are
    # We thereby assume that the label is not in any different column
    for column in data_frame:
        data_frame_label_subset = data_frame[data_frame[column] == label]
        # figure out which column start is and which one is end
        if maximum_length_dataframe < data_frame_label_subset.shape[0]:
            maximum_length_dataframe = data_frame_label_subset.shape[0]
            data_frame_label = data_frame_label_subset
        try:
            numerics.append((float(data_frame[column].iloc[0]), column))
        except:
            pass
    if maximum_length_dataframe == 0:
        return 0
    numerics.sort()
    start_column = numerics[0][1]
    end_column = numerics[1][1]
    return np.sum(data_frame_label[end_column].values - data_frame_label[start_column].values)



    # print(basename(csv_file)[:8] + ", " + label +  ": " + str(maximum_length_dataframe))
    return maximum_length_dataframe



def create_continuous_chunked_label_files(in_path, audio_length, out_dir, chunk_length, frame_length, detection_labels, keep_original_labels=False):
    number_of_chunks = int(audio_length // chunk_length)
    number_of_frames_per_chunk = int(np.around(chunk_length / frame_length))
    number_of_frames_session = int(np.around(audio_length / frame_length))
    leading_zeros_in_path = int(np.ceil(np.log10(number_of_chunks)))

    label_file_data_frame = pd.read_csv(in_path)
    label_file_iterator = label_file_data_frame.iterrows()


    current_row = next(label_file_iterator)
    current_start, current_end, current_label = current_row[1]["start"], current_row[1]["end"], current_row[1]["label"]
    continuous_labels_data_frame = pd.DataFrame(columns=["time", "label"])
    for current_frame in range(number_of_frames_session):
        current_time = current_frame * frame_length
        if current_time + 0.5 * frame_length > current_end:
            try:
                current_row = next(label_file_iterator)
                current_start, current_end, current_label = current_row[1]["start"], current_row[1]["end"], \
                                                            current_row[1]["label"]
            except:
                if keep_original_labels:
                    current_start = np.inf
                else:
                    break
        #if current_label == "child+adult":
        #    pass
        if current_time + 0.5 * frame_length > current_start and current_label in detection_labels:
            numeric_label = 1.
        else:
            numeric_label = 0.
        # overwrites numeric label with original label
        if keep_original_labels:
            if current_time + 0.5 * frame_length > current_start:
                numeric_label = current_label
            else:
                numeric_label = "None"
        continuous_labels_data_frame = continuous_labels_data_frame.append(
            pd.DataFrame([[current_time, numeric_label]], columns=["time", "label"]), ignore_index=True)
    number_of_chunks = int(len(continuous_labels_data_frame) * frame_length)
    for i in range(number_of_chunks):
        label_chunk_data_frame = continuous_labels_data_frame[i*number_of_frames_per_chunk : (i + 1) * number_of_frames_per_chunk]
        out_file_path_no_ext, file_ext = splitext(basename(in_path))
        out_file_path = out_dir + out_file_path_no_ext + "_" + str(i).zfill(leading_zeros_in_path) + file_ext
        label_chunk_data_frame.to_csv(out_file_path)

def get_features_from_file(csv_file, start_column=0, end_column=-1, delimiter=","):
    data_frame = pd.read_csv(csv_file, delimiter=get_delimeter(csv_file))
    return data_frame.iloc[:, start_column:end_column].values

def load_and_process_feature_single_file(feature_file, start_column=0, end_column=-1, delimiter=";", scaler=None):
    item_features = get_features_from_file(feature_file, start_column=start_column,
                                           delimiter=delimiter,
                                           end_column=end_column)
    if scaler != None:
        item_features = scaler.fit_transform(item_features)
    item_features = item_features[np.newaxis, ...]
    return item_features, item_features.shape[1]

def load_and_process_features_build_sequence(feature_files, start_column=0, end_column=-1, delimiter=";", scaler=None):
    features = []
    for feature_file in feature_files:
        item_features = get_features_from_file(feature_file, start_column=start_column,
                                           delimiter=delimiter,
                                           end_column=end_column)
        features.append(np.squeeze(item_features, axis=0))
    features = np.array(features)
    if scaler != None:
        features = scaler.fit_transform(features)
    features = features[np.newaxis, ...]
    return features, features.shape[1]

def get_num_lines_csv(file_name, skip_header=False):
    if skip_header:
        df = pd.read_csv(file_name, header=None)
    else:
        df = pd.read_csv(file_name)
    return df.shape[0]

def get_num_columns_csv(file_name):
    delim = get_delimeter(file_name)
    df = pd.read_csv(file_name, delimiter=delim)
    return df.shape[1]

def save_batch(data_batch, filepath_batch, out_dir):
    for data, filepath in zip(data_batch, filepath_batch):
        out_file_name_path = out_dir + splitext(basename(filepath))[0] + ".csv"
        np.savetxt(out_file_name_path, data, delimiter=",")