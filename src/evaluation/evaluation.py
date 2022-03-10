import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd
import tensorflow as tf
from src.utils.constants import *
from src.utils.plot_utils import plot_value_arrays
import matplotlib.pyplot as plt

def get_predictions_targets_torch(model, device, dataset):
    model.to(device)
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )
    outputs = torch.zeros((len(dataset), len(dataset.label_dict_int_to_string)))
    targets = torch.zeros((len(dataset)))
    with torch.no_grad():
        for index, (features, target) in tqdm(enumerate(loader), desc='Batch', total=len(loader)):
            # NOTE: again, squeeze might not be needed
            outputs[index, :] = model(features.to(device).squeeze(dim=1))['clipwise_output']
            targets[index] = torch.argmax(target)
    return outputs, targets

def accuracy_torch(outputs, targets):
    class_predictions = torch.argmax(outputs,1).float()
    return torch.sum(torch.eq(class_predictions, targets)).item()/targets.shape[0]


def confusion_matrix_from_torch(outputs, targets, metrics=[ACCURACY]):
    class_predictions = torch.argmax(outputs,1).float()
    num_classes = outputs.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(class_predictions)):
        prediction = class_predictions[i].int()
        correct = targets[i].int()
        confusion_matrix[prediction, correct] += 1
    return confusion_matrix

def accuracy(confusion_matrix):
    # confusion_matrix = np.array([[40,20,10], [10,10,10], [0,0,0]])
    return np.trace(confusion_matrix)/np.sum(confusion_matrix)

def uar(confusion_matrix):
    # confusion_matrix = np.array([[40, 20, 10], [10, 10, 10], [0, 0, 0]])
    TP_per_class = np.diag(confusion_matrix)
    P_per_class = np.sum(confusion_matrix, axis=0)
    recall_per_class = TP_per_class / P_per_class
    return recall_per_class.mean()

def uap(confusion_matrix):
    # confusion_matrix = np.array([[40, 20, 10], [10, 10, 10], [0, 0, 0]])
    TP_per_class = np.diag(confusion_matrix)
    pred_per_class = np.sum(confusion_matrix, axis=1)
    precision_per_class = TP_per_class / pred_per_class
    return precision_per_class.mean()

def uaf_one(confusion_matrix):
    # confusion_matrix = np.array([[40, 20, 10], [10, 10, 10], [0, 0, 0]])
    TP_per_class = np.diag(confusion_matrix)
    P_per_class = np.sum(confusion_matrix, axis=0)
    recall_per_class = TP_per_class / P_per_class
    pred_per_class = np.sum(confusion_matrix, axis=1)
    precision_per_class = TP_per_class / pred_per_class
    f_one_per_class = 2 * recall_per_class * precision_per_class / (precision_per_class + recall_per_class)
    return f_one_per_class.mean()

def evaluate_model_on_partition_torch(model, device, dataset, metrics=[CROSS_ENTROPY, ACCURACY]):
    outputs, targets = get_predictions_targets_torch(model, device, dataset)
    metric_value_dict = {}
    confusion_matrix = confusion_matrix_from_torch(outputs, targets)
    for metric in metrics:
        if metric == CROSS_ENTROPY:
            criterion = torch.nn.CrossEntropyLoss()
            metric_value_dict[metric] = criterion(outputs, targets.long())
        elif metric == ACCURACY:
            metric_value_dict[metric] = accuracy(confusion_matrix)
        elif metric == UAR:
            metric_value_dict[metric] = uar(confusion_matrix)
        elif metric == UAP:
            metric_value_dict[metric] = uap(confusion_matrix)
        elif metric == UAF_ONE:
            metric_value_dict[metric] = uaf_one(confusion_matrix)

    return outputs, targets, metric_value_dict



def evaluate_model_torch(model, device, train_dataset=None, devel_dataset=None, metrics=[CROSS_ENTROPY, ACCURACY]):
    train_outputs = None
    train_targets = None
    train_metric_dict = None
    if not train_dataset == None:
        train_outputs, train_targets, train_metric_dict = evaluate_model_on_partition_torch(model, device, train_dataset, metrics=metrics)
    devel_outputs = None
    devel_targets = None
    devel_metric_dict = None
    if not devel_dataset == None:
        devel_outputs, devel_targets, devel_metric_dict = evaluate_model_on_partition_torch(model, device, devel_dataset, metrics=metrics)
    return train_outputs, train_targets, train_metric_dict, devel_outputs, devel_targets, devel_metric_dict

def update_metric_dict(metric_dict, update_metric_dict):
    for key in update_metric_dict.keys():
        if key in metric_dict.keys():
            metric_dict[key].append(update_metric_dict[key])
        else:
            metric_dict[key] = [update_metric_dict[key]]
    return metric_dict

def update_metric_dicts_for_epoch(train_metric_dict=None, update_train_metric_dict=None, devel_metric_dict=None, update_devel_metric_dict=None):
    if train_metric_dict != None:
        train_metric_dict = update_metric_dict(train_metric_dict, update_train_metric_dict)
    if devel_metric_dict != None:
        devel_metric_dict = update_metric_dict(devel_metric_dict, update_devel_metric_dict)
    return train_metric_dict, devel_metric_dict

def store_results_to_text(result_path, train_metric_dict=None, devel_metric_dict=None, metrics = []):
    separation_str = "---------------------------------------------------------------------------------------------\n"
    result_lines = []
    #this might be solved more elegantly:
    epochs = 0
    if train_metric_dict != None:
        epochs = max(len(train_metric_dict[metrics[0]]), epochs)
    if devel_metric_dict != None:
        epochs = max(len(devel_metric_dict[metrics[0]]), epochs)
    for epoch in range(epochs):
        result_lines.append(separation_str)
        result_lines.append("Epoch \t" + str(epoch).zfill(3))
        result_lines.append(separation_str)
        if train_metric_dict != None:
            result_lines.append("Train\n")
            for metric in metrics:
                try:
                    result_lines.append(metric + ": {}\n".format(train_metric_dict[metric][epoch]))
                except:
                    pass
        result_lines.append(separation_str)
        if devel_metric_dict != None:
            result_lines.append("Devel\n")
            for metric in metrics:
                try:
                    result_lines.append(metric + ": {}\n".format(devel_metric_dict[metric][epoch]))
                except:
                    pass
        result_lines.append(separation_str)
    with open(result_path, "w") as f:
        f.writelines(result_lines)



def store_results(result_dir, train_metric_dict=None, devel_metric_dict=None, metrics = [], result_basename="result.txt"):
    store_results_to_text(result_dir + result_basename, train_metric_dict=train_metric_dict, devel_metric_dict=devel_metric_dict, metrics=metrics)
    for metric in metrics:
        labels = []
        value_arrays = []
        try:
            values = train_metric_dict[metric]
            value_arrays.append(values)
            labels.append("Train")
        except:
            pass
        try:
            values = devel_metric_dict[metric]
            value_arrays.append(values)
            labels.append("Devel")
        except:
            pass
        plot_value_arrays(result_dir + metric + ".png", value_arrays, labels=labels)





def get_tp_fp_detection(predictions, labels, threshold_step=0.001):
    true_positives = []
    false_positives = []
    # to be sure for rounding errors
    ground_truth = labels >= 0.99
    positives = np.sum(ground_truth == True)
    negatives = np.sum(ground_truth == False)
    for threshold in np.arange(0, 1+threshold_step, threshold_step):
        detections = predictions >= threshold
        true_positives.append(np.sum(np.where(detections == True,detections == ground_truth, False)))
        false_positives.append(np.sum(np.where(detections == True, detections != ground_truth, False)))
    return np.array(true_positives), np.array(false_positives), positives, negatives

def auc_detection_values_from_generator_iterative_approximation(model, datagenerator, threshold_step=0.001):
    true_positives = np.zeros(int(1 / threshold_step) + 1)
    false_positives = np.zeros(int(1 / threshold_step) + 1)
    positives = 0
    negatives = 0
    for i in range(len(datagenerator)):
        X, Y = datagenerator.__getitem__(i)
        predictions = model.predict(X)
        update_true_positives, update_false_positives, update_positives, update_negatives = get_tp_fp_detection(predictions, Y, threshold_step)
        true_positives += update_true_positives
        false_positives += update_false_positives
        positives += update_positives
        negatives += update_negatives
    tpr = true_positives/positives
    fpr = false_positives/negatives
    return fpr, tpr

def roc_curve_from_datagenerator(model, datagenerator):
    predictions_gathered = []
    labels_gathered = []
    for i in range(len(datagenerator)):
        X, Y = datagenerator.__getitem__(i)
        predictions = model.predict(X).flatten().tolist()
        labels = (Y.flatten() >= 0.99).tolist()
        predictions_gathered += predictions
        labels_gathered += labels
    predictions_gathered = np.array(predictions_gathered)
    labels_gathered = np.array(labels_gathered)
    fpr, tpr, thresholds = roc_curve(labels_gathered, predictions_gathered)
    return fpr, tpr, thresholds

def roc_metrics(fpr, tpr, thresholds=[]):
    # from stackoverflow with interpolation
    EER_threshold = None
    if len(thresholds) == 0:
        EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # from stackoverflow with interpolation
    else:
        fnr = 1 - tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EER_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

    AUC = auc(fpr, tpr)

    return EER, AUC, EER_threshold

def aggregate_detection_results(model, datagenerator, detection_threshold, gather_threshold=0.5, frame_length=0.01, inference=False):
    if inference:
        df = pd.DataFrame(columns=["start", "end", "prediction"])
    else:
        df = pd.DataFrame(columns=["start", "end", "prediction", "label", "correct"])
    start = 0
    for i in range(len(datagenerator)):
        X, Y = datagenerator.__getitem__(i)
        predictions = model.predict(X)
        if inference:
            for sample_predictions in predictions:
                end = start + (len(sample_predictions) - 1) * frame_length
                detections = sample_predictions >= detection_threshold
                sample_prediction = np.mean(detections) >= gather_threshold
                df.loc[len(df)] = [start, end, sample_prediction]
                start = end + frame_length
        else:
            for sample_predictions, sample_Y in zip(predictions,Y):
                end = start + (len(sample_predictions) - 1) * frame_length
                detections = sample_predictions >= detection_threshold
                sample_prediction = np.mean(detections) >= gather_threshold
                labels = (sample_Y.flatten() >= 0.99)
                sample_label = np.mean(labels) >= gather_threshold
                correct = sample_prediction == sample_label
                df.loc[len(df)] = [start, end, sample_prediction, sample_label, correct]
                start = end + frame_length
    if inference:
        return df, 0
    else:
        return df, np.mean(df["correct"].values)

def ccc_tf(x, y):
    x_mean = tf.math.reduce_mean(x)
    return x_mean
    # print("x_mean: " + str(x_mean))
    y_mean = tf.math.reduce_mean(y)
    # print("y_mean: " + str(y_mean))
    covariance = tf.math.reduce_mean((x - x_mean) * (y - y_mean))
    return covariance
    # print("covariance: " + str(covariance))
    #x_var = 1.0 / (float(len(x)) - 1.0) * tf.math.reduce_sum(
    #    (x - x_mean) ** 2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    x_var = tf.reduce_mean((x - x_mean)**2)
    # print("x_var: " + str(x_var))
    #y_var = 1.0 / (float(len(y)) - 1.0) * tf.math.reduce_sum((y - y_mean) ** 2)
    y_var = tf.reduce_mean((y - y_mean)**2)
    # print("x_var: " + str(x_var))
    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)
    # print("CCC: " + str(CCC))
    return CCC


def calc_scores(x, y):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE

    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = np.nanmean((x_mean - x) ** 2)
    y_var = np.nanmean((y_mean - y) ** 2)

    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)

    PCC = covariance / (x_std * y_std)

    RMSE = np.sqrt(np.nanmean((x - y) ** 2))

    scores = np.array([CCC, PCC, RMSE])

    return scores


def evaluate_data_generator(data_generator, model, save_dir, partition="test"):
    Y_all = []
    Y_pred_all = []
    outlines = [partition + "\n"]
    outlines.append("-------------------------------------------------\n")
    all_scores = []
    for i in range(len(data_generator)):
        X, Y = data_generator.__getitem__(i)
        Y_pred = model.predict(X)
        scores = calc_scores(Y, Y_pred)
        line = str(scores)
        print(line)
        outlines.append(line + "\n")
        all_scores.append(scores)
        Y_all.append(Y.flatten())
        Y_pred_all.append(Y_pred.flatten())
    all_scores = np.array(all_scores)
    line = "------------------------------------------"
    print(line)
    outlines.append(line + "\n")
    line = "Average: " + str(np.mean(all_scores,axis=0))
    print(line)
    outlines.append(line + "\n")
    Y_pred_all = np.hstack(Y_pred_all)
    Y_all = np.hstack(Y_all)
    scores = calc_scores(Y_all, Y_pred_all)
    outlines.append(line + "\n")
    line = "Final scores: " + str(scores)
    print(line)
    outlines.append(line + "\n")
    filename = save_dir + partition + ".txt"
    with open(filename, "w") as f:
        f.writelines(outlines)



def plot_data_generator_predictions(data_generator, model, save_dir, partition):
    for i in range(len(data_generator)):
        X, Y = data_generator.__getitem__(i)
        Y = Y.flatten()
        Y_pred = model.predict(X).flatten()
        x_axis_values = np.arange(Y_pred.shape[0])
        plt.plot(x_axis_values, np.squeeze(Y_pred), "g")
        plt.plot(x_axis_values, np.squeeze(Y), "r")
        plt.savefig(save_dir + partition + str(i) + ".png")
        plt.clf()

def save_history(history, save_dir, partition="Train"):
    history_path = save_dir + partition + ".csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False, header=False)