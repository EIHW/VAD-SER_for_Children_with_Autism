import matplotlib.pyplot as plt
import pandas as pd
from src.utils.csv_utils import get_delimeter
from glob import glob
from src.utils.constants import *
from src.utils.path_utils import get_system_dependendent_paths, make_directory
from glob import glob
import pandas as pd
from src.utils.plot_utils import make_nice_line_plot
from os.path import basename, splitext
import numpy as np
import matplotlib.pyplot as plt

nas_dir, code_dir = get_system_dependendent_paths(LOCAL)

all_labels_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/chunked_general_vad_1.0_labels/"
gt_all_labels_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/chunked_labels_all_1.0_labels/"
gt_child_labels_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/chunked_labels_1.0_labels/"
child_vad_labels_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/chunked_child_vad_1.0_labels/"
all_vad_labels_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/chunked_general_vad2_1.0_labels/"
webrtc_vad_labels_dir = "/home/manu/eihw/data_work/manuel/data/EMBOA/Valence_Arousal/chunked_webrtc_1.0_labels/"
test_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/test.csv"
train_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/train.csv"
devel_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/devel.csv"
image_dir = code_dir + "images_labels/"
make_directory(image_dir)
np_dir = code_dir + "numpy_histograms/"
make_directory(np_dir)
tasks = ["vads", "partitions"]
# tasks = ["partitions"]
# tasks = ["vads"]
mode = "load"
n_bins = 100
font_size = 16
figsize = (8, 6)

def gather_labels(label_dir, partition_file, np_dir, vad):
    print("-------------------------------------------------------------")
    print(basename(partition_file))
    print(label_dir)
    partition_info_df = pd.read_csv(partition_file, header=None)
    partition_info = partition_info_df.iloc[:, 0].values
    arousal_values = []
    valence_values = []
    arousal_filename = np_dir + vad + "_" + splitext(basename(partition_file))[0] + "_arousal.npy"
    valence_filename = np_dir + vad + "_" + splitext(basename(partition_file))[0] + "_valence.npy"
    for child_id in partition_info:
        if child_id[0] != "B":
            continue

        label_files = glob(label_dir + child_id + "*")
        label_files.sort()
        print(child_id + ": data points: {}".format(len(label_files)))
        if len(label_files) == 0:
            continue
        label_delimiter = get_delimeter(label_files[0])
        for label_file in label_files:
            label_df = pd.read_csv(label_file, delimiter=label_delimiter)
            arousal_values.append(label_df.iloc[0, 1])
            valence_values.append(label_df.iloc[0, 2])
    arousal_array = np.array(arousal_values)
    valence_array = np.array(valence_values)
    np.save(arousal_filename, arousal_array)
    np.save(valence_filename, valence_array)
    return np.histogram(arousal_array,  bins=100), np.histogram(valence_array,  bins=100)



#train_arousal_histogram, train_valence_histogram = gather_labels(label_dir, train_file)
if mode == "load":
    all_devel_arousal = np.load(np_dir + "all_devel_arousal.npy")
    all_devel_arousal_histogram = np.histogram(all_devel_arousal, bins=n_bins)
    all_test_arousal = np.load(np_dir + "all_test_arousal.npy")
    all_test_arousal_histogram = np.histogram(all_test_arousal, bins=n_bins)
    all_train_arousal = np.load(np_dir + "all_train_arousal.npy")
    all_train_arousal_histogram = np.histogram(all_train_arousal, bins=n_bins)
    gt_all_test_arousal = np.load(np_dir + "gt_all_test_arousal.npy")
    gt_all_test_arousal_histogram = np.histogram(gt_all_test_arousal, bins=n_bins)
    gt_child_test_arousal = np.load(np_dir + "gt_child_test_arousal.npy")
    gt_child_test_arousal_histogram = np.histogram(gt_child_test_arousal, bins=n_bins)
    all_vad_test_arousal = np.load(np_dir + "all_vad_test_arousal.npy")
    all_vad_test_arousal_histogram = np.histogram(all_vad_test_arousal, bins=n_bins)
    child_vad_test_arousal = np.load(np_dir + "child_vad_test_arousal.npy")
    child_vad_test_arousal_histogram = np.histogram(child_vad_test_arousal, bins=n_bins)
    webrtc_vad_test_arousal = np.load(np_dir + "webrtc_vad_test_arousal.npy")
    webrtc_vad_test_arousal_histogram = np.histogram(webrtc_vad_test_arousal, bins=n_bins)

    all_devel_valence = np.load(np_dir + "all_devel_valence.npy")
    all_devel_valence_histogram = np.histogram(all_devel_valence, bins=n_bins)
    all_test_valence = np.load(np_dir + "all_test_valence.npy")
    all_test_valence_histogram = np.histogram(all_test_valence, bins=n_bins)
    all_train_valence = np.load(np_dir + "all_train_valence.npy")
    all_train_valence_histogram = np.histogram(all_train_valence, bins=n_bins)
    gt_all_test_valence = np.load(np_dir + "gt_all_test_valence.npy")
    gt_all_test_valence_histogram = np.histogram(gt_all_test_valence, bins=n_bins)
    gt_child_test_valence = np.load(np_dir + "gt_child_test_valence.npy")
    gt_child_test_valence_histogram = np.histogram(gt_child_test_valence, bins=n_bins)
    all_vad_test_valence = np.load(np_dir + "all_vad_test_valence.npy")
    all_vad_test_valence_histogram = np.histogram(all_vad_test_valence, bins=n_bins)
    child_vad_test_valence = np.load(np_dir + "child_vad_test_valence.npy")
    child_vad_test_valence_histogram = np.histogram(child_vad_test_valence, bins=n_bins)
    webrtc_vad_test_valence = np.load(np_dir + "webrtc_vad_test_valence.npy")
    webrtc_vad_test_valence_histogram = np.histogram(webrtc_vad_test_valence, bins=n_bins)
else:
    all_test_arousal_histogram, all_test_valence_histogram = gather_labels(all_labels_dir, test_file, np_dir, "all")
if "vads" in tasks:
    if not mode == "load":
        gt_all_test_arousal_histogram, gt_all_test_valence_histogram = gather_labels(gt_all_labels_dir, test_file, np_dir, "gt_all")
        #np.save(np_dir + "all_test_arousal_histogram.npy", gt_all_test_arousal_histogram)
        #np.save(np_dir + "all_test_valence_histogram.npy", gt_all_test_valence_histogram)
        gt_child_test_arousal_histogram, gt_child_test_valence_histogram = gather_labels(gt_child_labels_dir, test_file, np_dir, "gt_child")
        child_vad_test_arousal_histogram, child_vad_test_valence_histogram = gather_labels(child_vad_labels_dir, test_file, np_dir, "child_vad")
        all_vad_test_arousal_histogram, all_vad_test_valence_histogram = gather_labels(all_vad_labels_dir, test_file, np_dir, "all_vad")
        webrtc_vad_test_arousal_histogram, webrtc_vad_test_valence_histogram = gather_labels(webrtc_vad_labels_dir, test_file, np_dir, "webrtc_vad")

    Xs = [all_test_arousal_histogram[1][:-1], gt_all_test_arousal_histogram[1][:-1], gt_child_test_arousal_histogram[1][:-1], child_vad_test_arousal_histogram[1][:-1], all_vad_test_arousal_histogram[1][:-1], webrtc_vad_test_arousal_histogram[1][:-1]]
    Ys = [ all_test_arousal_histogram[0] / np.sum(all_test_arousal_histogram[0]), gt_all_test_arousal_histogram[0] / np.sum(gt_all_test_arousal_histogram[0]), gt_child_test_arousal_histogram[0] / np.sum(gt_child_test_arousal_histogram[0]), child_vad_test_arousal_histogram[0] / np.sum(child_vad_test_arousal_histogram[0]), all_vad_test_arousal_histogram[0] / np.sum(all_vad_test_arousal_histogram[0]), webrtc_vad_test_arousal_histogram[0] / np.sum(webrtc_vad_test_arousal_histogram[0])]
    labels = ["All Audio", "All Vocalisations", "Child Vocalisations", "General VAD", "Child VAD", "WebRTC VAD"]
    image_path = image_dir + "arousal_plot_vad.jpg"
    x_axis = "Arousal"
    y_axis = "Frequency"
    title = "Label Distribution Test Partition (Arousal)"
    make_nice_line_plot(image_path, Xs, Ys, labels, font_size=font_size, x_axis=x_axis, y_axis=y_axis, title=title, fig_size=figsize)

    Xs = [all_test_valence_histogram[1][:-1], gt_all_test_valence_histogram[1][:-1],
          gt_child_test_valence_histogram[1][:-1], child_vad_test_valence_histogram[1][:-1],
          all_vad_test_valence_histogram[1][:-1], webrtc_vad_test_valence_histogram[1][:-1]]
    Ys = [all_test_valence_histogram[0] / np.sum(all_test_valence_histogram[0]),
          gt_all_test_valence_histogram[0] / np.sum(gt_all_test_valence_histogram[0]),
          gt_child_test_valence_histogram[0] / np.sum(gt_child_test_valence_histogram[0]),
          child_vad_test_valence_histogram[0] / np.sum(child_vad_test_valence_histogram[0]),
          all_vad_test_valence_histogram[0] / np.sum(all_vad_test_valence_histogram[0]),
          webrtc_vad_test_valence_histogram[0] / np.sum(webrtc_vad_test_valence_histogram[0])]
    labels = ["All Audio", "All Vocalisations", "Child Vocalisations", "General VAD", "Child VAD", "WebRTC VAD"]
    image_path = image_dir + "valence_plot_vad.jpg"
    x_axis = "Valence"
    y_axis = "Frequency"
    title = "Label Distribution Test Partition (Valence)"
    make_nice_line_plot(image_path, Xs, Ys, labels, font_size=font_size, x_axis=x_axis, y_axis=y_axis, title=title, fig_size=figsize)



    # plt.plot(all_test_arousal_histogram[1][:-1], all_test_arousal_histogram[0] / np.sum(all_test_arousal_histogram[0]),
    #          label="all audio")
    # plt.plot(gt_all_test_arousal_histogram[1][:-1],
    #          gt_all_test_arousal_histogram[0] / np.sum(gt_all_test_arousal_histogram[0]), label="all vocalisations")
    # plt.plot(gt_child_test_arousal_histogram[1][:-1],
    #          gt_child_test_arousal_histogram[0] / np.sum(gt_child_test_arousal_histogram[0]),
    #          label="child vocalisations")
    # plt.plot(child_vad_test_arousal_histogram[1][:-1],
    #          child_vad_test_arousal_histogram[0] / np.sum(child_vad_test_arousal_histogram[0]), label="child VAD")
    # plt.plot(all_vad_test_arousal_histogram[1][:-1],
    #          all_vad_test_arousal_histogram[0] / np.sum(all_vad_test_arousal_histogram[0]), label="general VAD")
    # plt.plot(webrtc_vad_test_arousal_histogram[1][:-1],
    #          webrtc_vad_test_arousal_histogram[0] / np.sum(webrtc_vad_test_arousal_histogram[0]), label="WebRTC VAD")
    # plt.legend()
    # plt.xlabel("arousal")
    # plt.ylabel("frequency")
    # plt.savefig(image_dir + "arousal_plot_vad.png")
    # plt.clf()

    # plt.plot(all_test_valence_histogram[1][:-1], all_test_valence_histogram[0] / np.sum(all_test_valence_histogram[0]),
    #          label="all audio")
    # plt.plot(gt_all_test_valence_histogram[1][:-1],
    #          gt_all_test_valence_histogram[0] / np.sum(gt_all_test_valence_histogram[0]), label="Without VAD")
    # plt.plot(gt_child_test_valence_histogram[1][:-1],
    #          gt_child_test_valence_histogram[0] / np.sum(gt_child_test_valence_histogram[0]),
    #          label="child vocalisations")
    # plt.plot(child_vad_test_valence_histogram[1][:-1],
    #          child_vad_test_valence_histogram[0] / np.sum(child_vad_test_valence_histogram[0]), label="child VAD")
    # plt.plot(all_vad_test_valence_histogram[1][:-1],
    #          all_vad_test_valence_histogram[0] / np.sum(all_vad_test_valence_histogram[0]), label="general VAD")
    # plt.plot(webrtc_vad_test_valence_histogram[1][:-1],
    #          webrtc_vad_test_valence_histogram[0] / np.sum(webrtc_vad_test_valence_histogram[0]), label="WebRTC VAD")
    # plt.legend()
    # plt.xlabel("valence")
    # plt.ylabel("frequency")
    #
    # plt.savefig(image_dir + "valence_plot_vad.png")
    # plt.clf()


if "partitions" in tasks:
    if not mode == "load":
        all_train_arousal_histogram, all_train_valence_histogram = gather_labels(all_labels_dir, train_file, np_dir, "all")
        all_devel_arousal_histogram, all_devel_valence_histogram = gather_labels(all_labels_dir, devel_file, np_dir, "all")


    Xs = [all_train_arousal_histogram[1][:-1], all_devel_arousal_histogram[1][:-1], all_test_arousal_histogram[1][:-1]]
    Ys = [all_train_arousal_histogram[0] / np.sum(all_train_arousal_histogram[0]), all_devel_arousal_histogram[0] / np.sum(all_devel_arousal_histogram[0]), all_test_arousal_histogram[0] / np.sum(all_test_arousal_histogram[0])]
    labels = ["Train", "Devel", "Test"]
    image_path = image_dir + "arousal_plot_partitions.jpg"
    x_axis = "Arousal"
    y_axis = "Frequency"
    title = "Label Distribution All Audio (Arousal)"
    make_nice_line_plot(image_path, Xs, Ys, labels, font_size=font_size, x_axis=x_axis, y_axis=y_axis, title=title, fig_size=figsize)

    Xs = [all_train_valence_histogram[1][:-1], all_devel_valence_histogram[1][:-1], all_test_valence_histogram[1][:-1]]
    Ys = [all_train_valence_histogram[0] / np.sum(all_train_valence_histogram[0]),
          all_devel_valence_histogram[0] / np.sum(all_devel_valence_histogram[0]),
          all_test_valence_histogram[0] / np.sum(all_test_valence_histogram[0])]
    labels = ["Train", "Devel", "Test"]
    image_path = image_dir + "valence_plot_partitions.jpg"
    x_axis = "Valence"
    y_axis = "Frequency"
    title = "Label Distribution All Audio (Valence)"
    make_nice_line_plot(image_path, Xs, Ys, labels, font_size=font_size, x_axis=x_axis, y_axis=y_axis, title=title, fig_size=figsize)

    #
    # plt.plot(all_train_arousal_histogram[1][:-1], all_train_arousal_histogram[0] / np.sum(all_train_arousal_histogram[0]),
    #          label="train")
    # plt.plot(all_devel_arousal_histogram[1][:-1], all_devel_arousal_histogram[0] / np.sum(all_devel_arousal_histogram[0]),
    #          label="devel")
    # plt.plot(all_test_arousal_histogram[1][:-1], all_test_arousal_histogram[0] / np.sum(all_test_arousal_histogram[0]),
    #          label="test")
    # plt.legend()
    # plt.xlabel("arousal")
    # plt.ylabel("frequency")
    # plt.savefig(image_dir + "arousal_plot_partitions.png")
    # plt.clf()
    #
    # plt.plot(all_train_valence_histogram[1][:-1],
    #          all_train_valence_histogram[0] / np.sum(all_train_valence_histogram[0]),
    #          label="train")
    # plt.plot(all_devel_valence_histogram[1][:-1],
    #          all_devel_valence_histogram[0] / np.sum(all_devel_valence_histogram[0]),
    #          label="devel")
    # plt.plot(all_test_valence_histogram[1][:-1], all_test_valence_histogram[0] / np.sum(all_test_valence_histogram[0]),
    #          label="test")
    # plt.legend()
    # plt.xlabel("valence")
    # plt.ylabel("frequency")
    # plt.savefig(image_dir + "valence_plot_partitions.png")
    # plt.clf()
    #







