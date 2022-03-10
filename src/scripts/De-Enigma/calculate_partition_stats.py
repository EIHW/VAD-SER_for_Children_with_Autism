from glob import glob
import pandas as pd
import numpy as np
from src.utils.wavfile_utils import read_wav_file_audiofile

audio_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/audio_normalised/"
diarisation_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/De-Enigma_Audio_Database/tier_0_diarisation/"
partition_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/data_split_files/"
crucial_labels_diarisation_all = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise", "therapist",
                      "therapist+zeno", "zeno"]
crucial_labels_diarisation_children = ["child", "child+therapist", "therapist+child", "child+zeno", "child+noise"]
partitions = ["train", "devel", "test"]
overall_total_time = 0
overall_total_children = 0
overall_total_sessions = 0
overall_total_children_speech = 0
overall_total_all_speech = 0
for partition in partitions:
    total_time = 0
    total_sessions = 0
    total_children_speech = 0
    total_all_speech = 0
    children_df = pd.read_csv(partition_dir + partition + ".csv")
    total_children = children_df.shape[0]
    children = np.squeeze(children_df.iloc[:].values)
    for child in children:
        diarisation_files = glob(diarisation_dir + child + "*")
        for diarisation_file in diarisation_files:
            diarisation_df = pd.read_csv(diarisation_file)
            for i in range(diarisation_df.shape[0]):
                label = diarisation_df.iloc[i, 2]
                if label in crucial_labels_diarisation_children:
                    total_children_speech += diarisation_df.iloc[i, 1] - diarisation_df.iloc[i, 0]
                if label in crucial_labels_diarisation_all:
                    total_all_speech += diarisation_df.iloc[i, 1] - diarisation_df.iloc[i, 0]

        audio_files = glob(audio_dir + child + "*_4*")
        audio_files.sort()
        total_sessions += len(audio_files)
        for audio_file in audio_files:
            signal, sr = read_wav_file_audiofile(audio_file)
            total_time += len(signal)/sr


    print("--------------------------------------")
    print(partition)
    print("Number of children: {}".format(total_children))
    print("Number of sessions: {}".format(total_sessions))
    print("total time in sec: {}".format(total_time))
    print("total time: {}:{}:{}".format(int(total_time/3600), int((total_time%3600)/60), total_time%60))
    print("total child speech: {}:{}:{}".format(int(total_children_speech / 3600), int((total_children_speech % 3600) / 60), total_children_speech % 60))
    print("total all speech: {}:{}:{}".format(int(total_all_speech / 3600), int((total_all_speech % 3600) / 60), total_all_speech % 60))
    overall_total_children_speech += total_children_speech
    overall_total_all_speech += total_all_speech
    overall_total_time += total_time
    overall_total_sessions += total_sessions
    overall_total_children += total_children
print("-------------------------------------")
print("overall")
print("Number of children: {}".format(overall_total_children))
print("Number of sessions: {}".format(overall_total_sessions))
print("total time in sec: {}".format(overall_total_time))
print("total time: {}:{}:{}".format(int(overall_total_time/3600), int((overall_total_time%3600)/60), overall_total_time%60))
print("total child speech: {}:{}:{}".format(int(overall_total_children_speech / 3600), int((overall_total_children_speech % 3600) / 60),
                                            overall_total_children_speech % 60))
print("total all speech: {}:{}:{}".format(int(overall_total_all_speech / 3600), int((overall_total_all_speech % 3600) / 60),
                                          overall_total_all_speech % 60))

