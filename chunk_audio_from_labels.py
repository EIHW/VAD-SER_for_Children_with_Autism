from src.utils.path_utils import make_directory
from src.utils.constants import *
from glob import glob
from os.path import basename, splitext
import pandas as pd
from src.utils.csv_utils import get_delimeter
import numpy as np


executing_PC = LOCAL
# vad_system = "webrtc"
# vad_system = "labels"
#vad_system = "labels_all"
# vad_system = "general_vad"
# vad_system = "general_vad2"
vad_system = "child_vad2"
#vad_system = "general_vad"

if vad_system == "child_vad" or vad_system == "general_vad" or vad_system == "child_vad2" or vad_system == "general_vad2":
    label_style = "prediction"
else:
    label_style = "annotation"

if vad_system == "child_vad" or vad_system == "child_vad2":
    eer_threshold = 0.235516518354415
elif vad_system == "general_vad" or vad_system == "general_vad2":
    eer_threshold = 0.528497695922851

detection_threshold = 0.25
#frames_per_second = 10
frames_per_second = 1

#frames_per_second = 100

feature_dir = "data_work/manuel/data/EMBOA/Valence_Arousal/egemaps_average_1/"
label_dir =  "data_work/manuel/data/EMBOA/Valence_Arousal/label_processed_1/"

if vad_system == "webrtc":
    vad_predictions_dir =  "data_work/manuel/data/EMBOA/Valence_Arousal/webrtcvad/"
elif vad_system == "labels" or vad_system == "labels_all":
    vad_predictions_dir = "data_work/manuel/data/EMBOA/De-Enigma_Audio_Database/tier_0_diarisation/"
elif vad_system == "child_vad":
    vad_predictions_dir = "data_work/manuel/data/EMBOA/VAD_child/predictionschild/"
elif vad_system == "general_vad":
    vad_predictions_dir = "data_work/manuel/data/EMBOA/VAD_child/predictionsall/"
elif vad_system == "child_vad2":
    vad_predictions_dir = "data_work/manuel/data/EMBOA/VAD_child/predictions_child2/"
elif vad_system == "general_vad2":
    vad_predictions_dir = "data_work/manuel/data/EMBOA/VAD_child/predictions_all2/"

out_feature_dir = "data_work/manuel/data/EMBOA/Valence_Arousal/chunked_" + vad_system + "_" + str(1/frames_per_second) + "_features/"
out_labels_dir = "data_work/manuel/data/EMBOA/Valence_Arousal/chunked_" + vad_system + "_" + str(1/frames_per_second) + "_labels/"
make_directory(out_labels_dir)
make_directory(out_feature_dir)

if vad_system == "labels":
    crucial_labels = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise"]
elif vad_system == "labels_all":
    crucial_labels = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise", "therapist",
                      "therapist+zeno", "zeno"]
elif vad_system == "webrtc":
    crucial_labels = ["speaker"]

label_files = glob(label_dir + "B*")
label_files.sort()
feature_files = glob(feature_dir + "B*")
feature_files.sort()


if label_style == "annotation":
    vad_predictions_files = glob(vad_predictions_dir + "B*")
    vad_predictions_files.sort()
    relevant_labels = vad_predictions_files[0]
    for label_file, feature_file, vad_predictions_file, in zip(label_files, feature_files, vad_predictions_files):
        child_session_id_label = splitext(basename(label_file))[0][:8]
        child_session_id_feature = splitext(basename(label_file))[0][:8]
        child_session_id_vad = splitext(basename(label_file))[0][:8]
        if child_session_id_feature != child_session_id_vad or child_session_id_label != child_session_id_feature:
            print("missmatch: " + child_session_id_vad + ", " + child_session_id_feature + ", " + child_session_id_label)
        print(splitext(basename(label_file))[0])
        label_df = pd.read_csv(label_file, delimiter=get_delimeter(label_file))
        feature_df = pd.read_csv(feature_file, delimiter=get_delimeter(feature_file))
        vad_predictions_df = pd.read_csv(vad_predictions_file)
        audio_len = feature_df.shape[0]/frames_per_second

        vad_df_iterator = vad_predictions_df.iterrows()
        current_row = next(vad_df_iterator)
        current_start, current_end, current_label = current_row[1]["start"], current_row[1]["end"], current_row[1]["label"]

        for time in range(int(audio_len)):
            crucial_label_time = 0
            while current_end < time + 1:
                if current_label in crucial_labels:
                    subtrahend = max(time, current_start)
                    minuend = min(time+1, current_end)
                    crucial_label_time += minuend - subtrahend
                try:
                    current_row = next(vad_df_iterator)
                    current_start, current_end, current_label = current_row[1]["start"], current_row[1]["end"], current_row[1][
                    "label"]
                except:
                    break
            if current_label in crucial_labels and time + 1 > current_start:
                subtrahend = max(time, current_start)
                minuend = time + 1
                crucial_label_time += minuend - subtrahend
            if crucial_label_time > detection_threshold:
                # do extraction stuff here!
                feature_df_chunk = feature_df.iloc[time * frames_per_second:(time + 1) * frames_per_second]
                label_df_chunk = label_df.iloc[time * frames_per_second: (time + 1) * frames_per_second]
                out_file_basename = child_session_id_vad + "_" + str(time).zfill(len(str(int(audio_len)))) + ".csv"
                feature_df_chunk.to_csv(out_feature_dir + out_file_basename, index=False)
                label_df_chunk.to_csv(out_labels_dir + out_file_basename, index=False)

elif label_style == "prediction":
    for feature_file, label_file in zip(feature_files, label_files):
        print(basename(feature_file))
        label_df = pd.read_csv(label_file, delimiter=get_delimeter(label_file))
        feature_df = pd.read_csv(feature_file, delimiter=get_delimeter(feature_file))
        base_filename = splitext(basename(feature_file))[0]
        vad_predictions_files = glob(vad_predictions_dir + base_filename + "*")
        vad_predictions_files.sort()
        for time, vad_prediction_file in enumerate(vad_predictions_files):
            prediction_df = pd.read_csv(vad_prediction_file, header=None)
            predictions = prediction_df.iloc[:,0].values
            detections = predictions > eer_threshold
            if np.mean(detections) > detection_threshold:
                if time < label_df.shape[0] and time < feature_df.shape[0]:
                    feature_df_chunk = feature_df.iloc[time * frames_per_second:(time + 1) * frames_per_second]
                    label_df_chunk = label_df.iloc[time * frames_per_second: (time + 1) * frames_per_second]
                    #out_file_basename = child_session_id_vad + "_" + str(time).zfill(len(str(int(audio_len)))) + ".csv"
                    out_file_basename = basename(vad_prediction_file)
                    feature_df_chunk.to_csv(out_feature_dir + out_file_basename, index=False)
                    label_df_chunk.to_csv(out_labels_dir + out_file_basename, index=False)
