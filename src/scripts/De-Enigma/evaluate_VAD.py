from src.utils.constants import *
from src.utils.path_utils import get_system_dependendent_paths, make_directory
from src.utils.csv_utils import get_delimeter
from glob import glob
import numpy as np
from os.path import splitext, basename

import pandas as pd

executing_PC = LOCAL
window_size = 0.01 #s
sequence_size = 1 #s
speaker = ALL

#vad_system = "label"
# vad_system = "all_vad"
vad_system = "child_vad"
#vad_system = "webrtc"

#eer_threshold = 0.528497695922851 # eer all run62
eer_threshold = 0.235516518354415 # eer child run59

if vad_system == "webrtc":
    vad_speakers = ["speaker"]
elif vad_system == "label":
    if speaker == CHILD:
        vad_speakers = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise"]
    elif speaker == ALL:
        vad_speakers = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise", "therapist",
                          "therapist+zeno", "zeno"]

if speaker == CHILD:
    label_speakers = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise"]
elif speaker == ALL:
    label_speakers = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise", "therapist",
                      "therapist+zeno", "zeno"]

nas_dir, code_dir = get_system_dependendent_paths(LOCAL)

label_dir = nas_dir + "data_work/manuel/data/EMBOA/De-Enigma_Audio_Database/tier_0_diarisation/"
if vad_system == "webrtc":
    vad_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/webrtcvad/"
elif vad_system == "label":
    vad_dir = nas_dir + "data_work/manuel/data/EMBOA/De-Enigma_Audio_Database/tier_0_diarisation/"
elif vad_system == "child_vad":
    vad_dir = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/predictions_child/"
elif vad_system == "all_vad":
    vad_dir = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/predictions_all/"


test_split_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/test.csv"
test_prefix_df = pd.read_csv(test_split_file, header=None)
prefixes = test_prefix_df.iloc[:,0].values


label_files = glob(label_dir + "*")
label_files.sort()


TPs = []
TNs = []
FPs = []
FNs = []
if vad_system == "webrtc" or vad_system == "label":
    vad_files = glob(vad_dir + "*")
    vad_files.sort()
    for label_file, vad_file in zip(label_files, vad_files):
        child_id = basename(label_file)[:4]
        if not child_id in prefixes or child_id[0] == "S":
            continue
        print(label_file)
        label_df = pd.read_csv(label_file, delimiter=get_delimeter(label_file))
        vad_df = pd.read_csv(vad_file, delimiter=get_delimeter(vad_file))
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        current_label_idx = 0
        current_start_label = label_df.iloc[current_label_idx]["start"]
        current_end_label = label_df.iloc[current_label_idx]["end"]
        current_interval_label = label_df.iloc[current_label_idx]["label"]
        current_vad_idx = 0
        current_start_vad = vad_df.iloc[current_vad_idx]["start"]
        current_end_vad = vad_df.iloc[current_vad_idx]["end"]
        current_interval_vad = vad_df.iloc[current_vad_idx]["label"]
        final_time_label = label_df.iloc[-1]["end"]
        final_time_vad = vad_df.iloc[-1]["end"]
        final_time = max(final_time_vad, final_time_label)

        for time in np.arange(0, final_time, window_size):
            if time > current_end_label and current_label_idx < label_df.shape[0]-1:
                #print("label indx" + str(current_label_idx))
                current_label_idx += 1
                current_start_label = label_df.iloc[current_label_idx]["start"]
                current_end_label = label_df.iloc[current_label_idx]["end"]
                current_interval_label = label_df.iloc[current_label_idx]["label"]
            if time > current_end_vad and current_vad_idx < vad_df.shape[0] - 1:
                current_vad_idx += 1
                #print("vad-idx" + str(current_vad_idx))
                current_start_vad = vad_df.iloc[current_vad_idx]["start"]
                current_end_vad = vad_df.iloc[current_vad_idx]["end"]
                current_interval_vad = vad_df.iloc[current_vad_idx]["label"]

            if time >= current_start_label and time <= current_end_label and current_interval_label in label_speakers:
                label = True
            else:
                label = False
            if time >= current_start_vad and time <= current_end_vad and current_interval_vad in vad_speakers:
                prediction = True
            else:
                prediction = False
            if label:
                if prediction:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction:
                    FP += 1
                else:
                    TN += 1
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)
        TNs.append(TN)
        print("TP: {}, FP: {}, FN: {}, TN: {}".format(TP, FP, FN, TN))
        try:
            print("precision: {}, recall: {}, FPR: {}".format(TP/(TP+FP), TP/(TP+FN), FP/(FP+TN)))
        except:
            print("zero division!")
else:
    for label_file in label_files:
        child_session_id = basename(label_file)[:8]
        child_id = basename(label_file)[:4]
        if not child_id in prefixes or child_id[0] == "S":
            continue
        label_df = pd.read_csv(label_file, delimiter=get_delimeter(label_file))
        print(label_file)
        vad_files = glob(vad_dir + child_session_id + "*")
        vad_files.sort()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        current_label_idx = 0
        current_start_label = label_df.iloc[current_label_idx]["start"]
        current_end_label = label_df.iloc[current_label_idx]["end"]
        current_interval_label = label_df.iloc[current_label_idx]["label"]

        vad_file_idx = 0
        final_time_label = label_df.iloc[-1]["end"]
        final_time_vad = len(vad_files) - 1
        final_time = max(final_time_vad, final_time_label)
        for time in np.arange(0, final_time, window_size):
            vad_df_idx = int(time/window_size) % int(sequence_size/window_size)
            if time > current_end_label and current_label_idx < label_df.shape[0]-1:
                #print("label indx" + str(current_label_idx))
                current_label_idx += 1
                current_start_label = label_df.iloc[current_label_idx]["start"]
                current_end_label = label_df.iloc[current_label_idx]["end"]
                current_interval_label = label_df.iloc[current_label_idx]["label"]
            if vad_df_idx == 0:
                #print(vad_file_idx)
                try:
                    vad_df = pd.read_csv(vad_files[vad_file_idx], header=None)
                    vad_file_idx += 1
                except:
                    vad_df.iloc[:] = np.zeros((100, 1))
            if time >= current_start_label and time <= current_end_label and current_interval_label in label_speakers:
                label = True
            else:
                label = False
            vad_prediction = vad_df.iloc[vad_df_idx].values[0]
            if vad_prediction >= eer_threshold:
                prediction = True
            else:
                prediction = False
            if label:
                if prediction:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction:
                    FP += 1
                else:
                    TN += 1
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)
        TNs.append(TN)
        print("TP: {}, FP: {}, FN: {}, TN: {}".format(TP, FP, FN, TN))
        try:
            print("precision: {}, recall: {}, FPR: {}".format(TP / (TP + FP), TP / (TP + FN), FP / (FP + TN)))
        except:
            print("zero division!")



print("------------------------------------------------------------")
print("overall: ")
print("precision: {}, recall: {}, FPR: {}".format(np.sum(TPs)/(np.sum(TPs)+np.sum(FPs)), np.sum(TPs)/(np.sum(TPs)+np.sum(FNs)), np.sum(FPs)/(np.sum(FPs)+np.sum(TNs))))
