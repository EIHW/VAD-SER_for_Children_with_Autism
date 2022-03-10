from src.data_generators.universal_generators import  Universal_Generator_from_Partition_File
from src.models.audio_feature_models import get_detection_model
from src.evaluation.evaluation import roc_curve_from_datagenerator, roc_metrics, aggregate_detection_results
from src.utils.plot_utils import simple_line_plot
import tensorflow as tf
import pandas as pd
import numpy as np
from os.path import splitext
from src.utils.constants import *
from src.utils.path_utils import new_run_directory, make_directory
from src.training.callbacks import Train_Call_Back
from src.utils.model_utils import infere_write_from_data_generator


mode = TRAIN_AND_EVALUATE
database = DE_ENIGMA
speaker = ALL
executing_pc = LOCAL
training = FROM_SCRATCH


print("Mode: " + mode)
print("Database: " + database)
print("Speaker: " + speaker)
print("Executing on: " + executing_pc)

#quick_test = True
epochs = 10
batch_size = 256
hop_size = 1
files_per_sample=1
validation = True

culture_limitations = None
age_limit = None
#age_limit = 9

microphones = [4]
feature_type_lld = "LLD"
feature_type_none = ""
feature_type = ""
# speaker = "all"


data_root_dir = "data_work/manuel/data/EMBOA/VAD_child/"

train_data_path = ""
test_data_path = ""
devel_data_path = ""


# data split files
train_data_path = data_root_dir + "data_split_files/train.csv"
test_data_path = data_root_dir + "data_split_files/test.csv"
devel_data_path = data_root_dir + "data_split_files/devel.csv"

#feature label and prediction files
feature_dir = data_root_dir + "normalised_data/"
label_dir = data_root_dir + "chunked_labels_1_0.01_" + speaker + "/"
prediction_dir = data_root_dir + "predictions_" + speaker + "/"

original_label_dir = data_root_dir + "chunked_labels_1_0.01_original/"
result_dir = "src/results/EMBOA_VAD/"

run_dir, current_run = new_run_directory(result_dir, mode)
make_directory(result_dir + run_dir)
checkpoint_folder = "src/models/trained/EMBOA_VAD/"
make_directory(checkpoint_folder)
checkpoint_folder += run_dir
make_directory(checkpoint_folder)

result_text_log_path = result_dir + run_dir + "result_log.txt"
history_path = result_dir + run_dir +"history.csv"

pre_trained_path = "src/models/trained/EMBOA_VAD/run_062/model_009.h5"

if mode == INFERENCE or mode == APPLY_MODEL:
    model_path = pre_trained_path
elif mode == EVALUATE or mode == EVALUATE_INDIVIDUALLY:
    model_path = pre_trained_path
else:
    model_path = "src/models/trained/EMBOA_VAD/" + run_dir.split("/")[-2] + "_model.h5"

result_lines = []

feature_normaliser_path = data_root_dir + "train_minxmax_scaler.pkl"
if mode != INFERENCE:
    feature_normaliser_path = None

elif mode == TRAIN_AND_EVALUATE or mode == TRAIN:
    train_data_generator = Universal_Generator_from_Partition_File(train_data_path, feature_dir, label_dir,
                                                                   batch_size=batch_size, data_limitations=culture_limitations, microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, feature_type=feature_type_none)
    if validation == True:
        devel_data_generator = Universal_Generator_from_Partition_File(devel_data_path, feature_dir, label_dir,
                                                                   batch_size=batch_size, data_limitations=culture_limitations, microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, feature_type=feature_type_none)
    else:
        devel_data_generator = None

    test_data_generator = Universal_Generator_from_Partition_File(test_data_path, feature_dir, label_dir, batch_size=batch_size, data_limitations=culture_limitations,microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, feature_type=feature_type_none)


elif mode == APPLY_MODEL:
    test_data_generator = Universal_Generator_from_Partition_File(test_data_path, feature_dir, label_dir,
                                                                  batch_size=batch_size,
                                                         data_limitations=culture_limitations, database=database, shuffle=False, hop_size=hop_size, files_per_sample=files_per_sample, feature_type=feature_type, microphones=microphones,
                                                                  normaliser_path=feature_normaliser_path, inference=True)
elif mode == EVALUATE:
    test_data_generator = Universal_Generator_from_Partition_File(test_data_path, feature_dir, label_dir,
                                                                  batch_size=batch_size,
                                                                  data_limitations=culture_limitations,
                                                                  database=database, shuffle=False,
                                                                  hop_size=hop_size, files_per_sample=files_per_sample,
                                                                  feature_type=feature_type,
                                                                  normaliser_path=feature_normaliser_path,
                                                                  inference=False, first_feature_column=2, last_feature_column=134, start_label_column=2, end_column_label=4)
#callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_folder, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callback = Train_Call_Back(checkpoint_folder)

if mode == INFERENCE or mode == EVALUATE or mode == EVALUATE_INDIVIDUALLY or training == CONTINUE or mode == APPLY_MODEL:
    model = tf.keras.models.load_model(model_path)
elif training == FROM_SCRATCH:
    model = get_detection_model(train_data_generator.feature_shape)



if mode == QUICK_TEST:
    # without devel
    history = model.fit(train_data_generator, epochs=3)#, validation_data=devel_data_generator, callbacks=[callback])
elif mode == TRAIN_AND_EVALUATE or mode == TRAIN:
    history = model.fit(train_data_generator, epochs=epochs, validation_data=devel_data_generator, callbacks=[callback])


if not (mode == INFERENCE or mode == APPLY_MODEL or mode == EVALUATE or mode == EVALUATE_INDIVIDUALLY):
    fpr, tpr, EER_Threshold = roc_curve_from_datagenerator(model, train_data_generator)
    simple_line_plot(fpr, tpr, result_dir + run_dir + "auc_after_overfit.png")
    EER, AUC, EER_threshold = roc_metrics(fpr, tpr)
    line = "After overfit: \t\t\t EER: {},\t\t AUC:{}, EER threshold: {}".format(EER, AUC, EER_threshold)
    print(line)
    result_lines.append(line + "\n")

elif mode == APPLY_MODEL:
    if database == DE_ENIGMA:
        #test_data_generator = infere_write_from_data_generator(test_data_generator, "")
        infere_write_from_data_generator(test_data_generator, model, prediction_dir)
        # add_original_labels_to_csv_file(result_dir + run_dir, original_label_dir, test_data_path)

fpr, tpr, thresholds = roc_curve_from_datagenerator(model, test_data_generator)
np.save(result_dir + run_dir + "fpr.npy", fpr)
np.save(result_dir + run_dir + "tpr.npy", tpr)
np.save(result_dir + run_dir + "thresholds.npy", thresholds)
simple_line_plot(fpr, tpr, result_dir + run_dir + "auc_after.png")
EER, AUC, EER_threshold = roc_metrics(fpr, tpr, thresholds=thresholds)
line = "After: \t\t\t EER: {},\t\t AUC:{}, EER threshold: {}".format(EER, AUC, EER_threshold)
print(line)
result_lines.append(line + "\n")




with open(result_text_log_path, "w") as f:
    f.writelines(result_lines)


if not (mode == INFERENCE or mode == EVALUATE or mode == EVALUATE_INDIVIDUALLY):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path)
    # model.save(model_path)


print("done")