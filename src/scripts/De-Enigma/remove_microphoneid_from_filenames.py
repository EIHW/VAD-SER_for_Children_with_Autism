from glob import glob
from os.path import basename, splitext
from os import rename

data_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/"
# child_feature_dir = data_dir + "chunked_child_vad_1.0_features/"
# child_label_dir = data_dir + "chunked_child_vad_1.0_labels/"
# all_feature_dir = data_dir + "chunked_general_vad_1.0_features/"
# all_label_dir = data_dir + "chunked_general_vad_1.0_labels/"
# all_feature_dir = data_dir + "chunked_general_vad2_1.0_features/"
# all_label_dir = data_dir + "chunked_general_vad2_1.0_labels/"
all_feature_dir = data_dir + "chunked_child_vad2_1.0_features/"
all_label_dir = data_dir + "chunked_child_vad2_1.0_labels/"


def remove_microphoneid_from_filename(dir):
    files = glob(dir + "*")
    files.sort()
    print(len(files))
    for file in files:
        file_split = basename(file).split("_")
        if len(file_split[-2]) == 1:
            del file_split[-2]
            out_file = dir + "_".join(file_split)
            rename(file, out_file)


# remove_microphoneid_from_filename(child_label_dir)
# remove_microphoneid_from_filename(child_feature_dir)
remove_microphoneid_from_filename(all_feature_dir)
remove_microphoneid_from_filename(all_label_dir)