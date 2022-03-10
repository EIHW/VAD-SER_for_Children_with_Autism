import pandas as pd
import numpy as np
from src.utils.csv_utils import get_delimeter
from src.utils.classification_utils import convert_labels_string_to_int
from src.utils.deenigma_utils import convert_deenigma_chunked_feature_path_to_label_path, convert_feature_path_to_label_path
from src.utils.constants import *
from glob import glob
from os.path import exists, basename

TRAIN = "train"

def get_audio_files_and_class_labels_from_csv(partition_file, quick_test=False, label_dict_int_to_string=None):
    data_frame = pd.read_csv(partition_file, delimiter=get_delimeter(partition_file))
    X = data_frame.iloc[:,0].values
    labels_as_strings = data_frame.iloc[:,1].values
    Y, label_dict_int_to_string = convert_labels_string_to_int(labels_as_strings, one_hot=True, label_dict_int_to_string=label_dict_int_to_string)
    if quick_test:
        return X[:32], Y[:32], label_dict_int_to_string
    #return X[:512], Y[:512], label_dict_int_to_string
    else:
        return X, Y, label_dict_int_to_string
# TODO: Add option about microphones!!! or at least check what's going on!
def get_feature_label_files_deenigma(partition_file, feature_dir=None, label_dir=None, feature_type="LLD", only_one_batch = False, data_limitations = None, database=None, inference=False, microphones=[], task_mode=MANY_TO_ONE):
    if partition_file != "":
        data_frame = pd.read_csv(partition_file, header=None)
        # This part is quite specific to de-enigma organisation.
        data_identifications = data_frame.iloc[:, 0].values
    else:
        data_identifications = [""]
    feature_files = []
    label_files = []
    # get feature files
    if feature_dir != None:
        for data_identification in data_identifications:
            # check limitations, i.e., if there is only one culture.
            if data_limitations != None:
                included = False
                # TODO: if data_identification[:len(limitation)] in data_limitations????
                for limitation in data_limitations:
                    if data_identification[:len(limitation)] == limitation:
                        included = True
                        break
                if not included:
                    continue
            if len(microphones) > 0:
                for microphone in microphones:
                    identification_wildcard = feature_dir + data_identification + "*" + "_" + str(microphone) + "*" + feature_type + "*"
                    if task_mode == BUILD_SEQUENCE:
                        new_feature_files = glob(identification_wildcard)
                        # TODO: Sequence building only supports it if we have a label file for each feature_file
                        if len(new_feature_files) != 0:
                            new_feature_files.sort()
                            feature_files.append(new_feature_files)
                            new_label_files = []
                            for feature_file in new_feature_files:
                                label_file = convert_deenigma_chunked_feature_path_to_label_path(feature_file, label_dir)
                                new_label_files.append(label_file)
                            label_files.append(new_label_files)

                    else:
                        feature_files += glob(identification_wildcard)

            else:
                identification_wildcard = feature_dir + data_identification + "*" + feature_type + "*"
                if task_mode == BUILD_SEQUENCE:
                    new_feature_files = glob(identification_wildcard)
                    # TODO: Sequence building only supports it if we have a label file for each feature_file
                    if len(new_feature_files) != 0:
                        new_feature_files.sort()
                        feature_files.append(new_feature_files)
                        new_label_files = []
                        for feature_file in new_feature_files:
                            label_file = convert_deenigma_chunked_feature_path_to_label_path(feature_file, label_dir)
                            new_label_files.append(label_file)
                        label_files.append(new_label_files)
                else:
                    feature_files += glob(identification_wildcard)
            # TODO: this is to have quick feature generation!
            #break
    if not task_mode == BUILD_SEQUENCE:
        feature_files.sort()
        # get label files
        feature_files_without_labels = []

        if label_dir != None and not inference:
            for feature_file in feature_files:
                if database == "EMBOA":
                    label_file = convert_feature_path_to_label_path(feature_file, label_dir)
                else:
                    label_file = convert_deenigma_chunked_feature_path_to_label_path(feature_file, label_dir)
                if exists(label_file):
                    label_files.append(label_file)
                else:
                    feature_files_without_labels.append(feature_file)
            feature_files = [x for x in feature_files if x not in feature_files_without_labels]
        else:
            label_files = np.array([])

    if only_one_batch:
        #return np.array(feature_files), np.array(label_files)
        # return np.array(feature_files)[:128], np.array(label_files)[:128]
        #return np.array(feature_files)[:1024], np.array(label_files)[:1024]
        return np.array(feature_files), np.array(label_files)

    else:
        # return np.array(feature_files)[:128], np.array(label_files)[:128]
        #return np.array(feature_files)[:1024], np.array(label_files)[:1024]
        if task_mode == BUILD_SEQUENCE:
            #return feature_files[:1], label_files[:1]
            return feature_files, label_files
        else:
            return np.array(feature_files), np.array(label_files)

def get_feature_label_files_deenigma_ser(partition_file, feature_dir=None, label_dir=None, feature_type="LLD", only_one_batch = False, data_limitations = None, database=None, inference=False):

    data_frame = pd.read_csv(partition_file)
    # This part is quite specific to de-enigma organisation.
    data_identifications = data_frame.iloc[:,0].values
    feature_files = []
    # get feature files
    if feature_dir != None:
        for data_identification in data_identifications:
            # check limitations, i.e., if there is only one culture.
            if data_limitations != None:
                included = False
                for limitation in data_limitations:
                    if data_identification[:len(limitation)] == limitation:
                        included = True
                        break
                if not included:
                    continue
            identification_wildcard = feature_dir + data_identification + "*" + feature_type + "*"
            feature_files += glob(identification_wildcard)
    feature_files.sort()
    # get label files
    feature_files_without_labels = []
    if label_dir != None:
        label_files = []
        for feature_file in feature_files:
            if database == "EMBOA":
                label_file = convert_feature_path_to_label_path(feature_file, label_dir)
            else:
                label_file = convert_deenigma_chunked_feature_path_to_label_path(feature_file, label_dir)
            if exists(label_file):
                label_files.append(label_file)
            else:
                feature_files_without_labels.append(feature_file)
    if inference:
        label_files = np.array([])
    else:
        feature_files = [x for x in feature_files if x not in feature_files_without_labels]
    if only_one_batch:
        return np.array(feature_files), np.array(label_files)
        #return np.array(feature_files)[:32], np.array(label_files)[:32]
    else:
        return np.array(feature_files), np.array(label_files)




def check_overlapping_files(partition_file_1, partition_file_2):
    data_frame_1 = pd.read_csv(partition_file_1, delimiter=get_delimeter(partition_file_1))
    X_1 = data_frame_1.iloc[:, 0].values
    X_1_set = set(X_1)
    print("Length of training set: {}".format(len(X_1_set)))
    data_frame_2 = pd.read_csv(partition_file_2, delimiter=get_delimeter(partition_file_2))
    X_2 = data_frame_2.iloc[:, 0].values
    X_2_set = set(X_2)
    print("Length of development set: {}".format(len(X_2_set)))
    intersect = X_1_set.intersection(X_2_set)
    print("Length of intersection: {}".format(len(intersect)))
    return intersect