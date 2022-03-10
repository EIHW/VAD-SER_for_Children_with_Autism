from src.utils.constants import *
from src.utils.path_utils import get_system_dependendent_paths, make_directory
from glob import glob
import pandas as pd
from os.path import basename, splitext
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error as mse

exec_pc = LOCAL
nas_dir, code_dir = get_system_dependendent_paths(exec_pc)

label_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/chunked_labels_0.1_labels/"
test_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/test.csv"
train_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/train.csv"
devel_file = nas_dir + "data_work/manuel/data/EMBOA/VAD_child/data_split_files/devel.csv"
image_dir = code_dir + "images/"
label_files = glob(label_dir + "*")
label_files.sort()
make_directory(image_dir)

def mse(x,y):
    return np.mean((x-y)**2)

def calc_stats(partition_file, label_files):
    print("-------------------------------------------------------------")
    print(partition_file)
    partition_info_df = pd.read_csv( partition_file, header=None)
    partition_info = partition_info_df.iloc[:,0].values
    arousal_labels = []
    last_arousal_labels = []
    valence_labels = []
    last_valence_labels = []
    for label_file in label_files:
        child_id = basename(label_file)[:4]
        if child_id not in partition_info:
            continue
        label_df = pd.read_csv(label_file)
        arousal_labels.append(label_df.iloc[:,1].values)
        last_arousal_labels.append(label_df.iloc[-1,1])
        valence_labels.append(label_df.iloc[:, 2].values)
        last_valence_labels.append(label_df.iloc[-1, 2])
        #break

    arousal_histogram = np.histogram(arousal_labels, bins=100)
    valence_histogram = np.histogram(valence_labels, bins=100)
    print("arousal mean: {}, arousal std: {}".format(np.mean(arousal_labels), np.std(arousal_labels)))
    print("valence mean: {}, valence std: {}".format(np.mean(valence_labels), np.std(valence_labels)))
    return arousal_histogram, valence_histogram, arousal_labels, valence_labels, last_arousal_labels, last_valence_labels

train_arousal_histogram, train_valence_histogram, train_arousal_labels, train_valence_labels, train_last_arousal_labels, train_last_valence_labels = calc_stats(train_file, label_files)
devel_arousal_histogram, devel_valence_histogram, devel_arousal_labels, devel_valence_labels, devel_last_arousal_labels, devel_last_valence_labels = calc_stats(devel_file, label_files)
test_arousal_histogram, test_valence_histogram , test_arousal_labels, test_valence_labels, test_last_arousal_labels, test_last_valence_labels = calc_stats(test_file, label_files)

plt.plot(train_arousal_histogram[1][:-1], train_arousal_histogram[0]/np.sum(train_arousal_histogram[0]), label="train")
plt.plot(devel_arousal_histogram[1][:-1], devel_arousal_histogram[0]/np.sum(devel_arousal_histogram[0]), label="devel")
plt.plot(test_arousal_histogram[1][:-1], test_arousal_histogram[0]/np.sum(test_arousal_histogram[0]), label="test")
plt.legend()
plt.savefig(image_dir + "arousal_plot.png")
plt.clf()

plt.plot(train_valence_histogram[1][:-1], train_valence_histogram[0]/np.sum(train_valence_histogram[0]), label="train")
plt.plot(devel_valence_histogram[1][:-1], devel_valence_histogram[0]/np.sum(devel_valence_histogram[0]), label="devel")
plt.plot(test_valence_histogram[1][:-1], test_valence_histogram[0]/np.sum(test_valence_histogram[0]), label="test")
plt.legend()
plt.savefig(image_dir + "valence_plot.png")
plt.clf()

print("----------------------------------------")
print("Arousal Evaluation:")
train_last_arousal_label_mean = np.mean(train_last_arousal_labels)
print("Chance MSE arousal: {}".format(mse(np.array(test_last_arousal_labels), train_last_arousal_label_mean)))

print("Valence Evaluation:")
train_last_valence_label_mean = np.mean(train_last_valence_labels)
print("Chance MSE valence: {}".format(mse(test_last_valence_labels, train_last_valence_label_mean)))