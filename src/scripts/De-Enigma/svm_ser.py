from src.utils.constants import *
from src.utils.path_utils import get_system_dependendent_paths, make_directory
from glob import glob
from src.utils.csv_utils import get_delimeter
import pandas as pd
from sklearn.svm import SVR
import numpy as np
from src.evaluation.evaluation import calc_scores
import matplotlib.pyplot as plt

executing_pc = LOCAL
label = "valence"

def mse(x,y):
    return np.mean((x-y)**2)

nas_dir, code_dir = get_system_dependendent_paths(executing_pc)
data_root_dir = nas_dir + "data_work/manuel/data/EMBOA/Valence_Arousal/"
feature_file = data_root_dir + "chunked_labels_1.0_features_all.csv"
label_file = data_root_dir + "chunked_labels_1.0_labels_all.csv"

feature_df = pd.read_csv(feature_file)
label_df = pd.read_csv(label_file)

# Very preliminary split (has to be replaced)
train_split = 0.8
features = feature_df.iloc[:,:-1].values
labels = label_df.iloc[:,2].values

print("correlations")
#for i in range(features.shape[1]):
#    print(np.corrcoef(features[:,i], labels))
print("---------------------------------------------")
train_features = features[:int(train_split*features.shape[0])]
test_features = features[int(train_split*features.shape[0]):]

train_labels = labels[:int(train_split*features.shape[0])]
test_labels = labels[int(train_split*features.shape[0]):]



plot_size = 256
for exp in np.arange(-1,5):
    C = 10.**(-exp)
    print("-------------------------------------------")
    svr = SVR(C=C)
    svr.fit(train_features, train_labels)
    print("Train")
    y_pred = svr.predict(train_features)
    print("MSE: " + str(mse(train_labels, y_pred)))
    print("scores: " + str(calc_scores(y_pred, train_labels)))
    plt.plot(np.arange(train_labels.shape[0])[:plot_size], train_labels[:plot_size], "r")
    plt.plot(np.arange(train_labels.shape[0])[:plot_size], y_pred[:plot_size], "b")
    plt.savefig("images/svr_" + str(C) + "train.png")
    plt.clf()

    print("Test")
    y_pred = svr.predict(test_features)
    print("MSE: " + str(mse(test_labels, y_pred)))
    print("scores: " + str(calc_scores(y_pred, test_labels)))
    plt.plot(np.arange(test_labels.shape[0])[:plot_size], test_labels[:plot_size], "r")
    plt.plot(np.arange(test_labels.shape[0])[:plot_size], y_pred[:plot_size], "b")
    plt.savefig("images/svr_" + str(C) + "test.png")
    plt.clf()

