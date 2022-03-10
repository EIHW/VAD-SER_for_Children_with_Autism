import pandas as pd
from src.data_generators.universal_generators import  Universal_Generator_from_Partition_File, deenigma_ser_generator
from src.models.audio_feature_models import *
import tensorflow as tf
import matplotlib.pyplot as plt
from src.evaluation.evaluation import calc_scores
from tensorflow.keras.optimizers import Adam
from src.evaluation.evaluation import ccc_tf
from src.evaluation.losses import ccc_loss_multiple_targets
from os.path import splitext
from src.utils.constants import *
from src.utils.path_utils import new_run_directory, make_directory
from src.utils.deenigma_utils import add_original_labels_to_csv_file
from src.training.callbacks import Train_Call_Back
from src.utils.model_utils import save_hyperparameters
from src.evaluation.evaluation import evaluate_data_generator, plot_data_generator_predictions, save_history
import numpy as np


mode = EVALUATE
# mode = TRAIN_AND_EVALUATE
database = DE_ENIGMA
training = FROM_SCRATCH
data_loader_mode = LOAD_ALL
label_name = "arousal"
# label_name = "valence"
if label_name == "arousal":
    start_label_column = 1
    end_label_column = 2
elif label_name == "valence":
    start_label_column = 2
    end_label_column = 3

print("Mode: " + mode)
print("Database: " + database)
#print("Speaker: " + speaker)
print(label_name)

#quick_test = True
epochs = 200
batch_size = 512
optimizer = "adam"
hop_size = 1
evaluation_batch_size = 1
model_callback_interval = 20
learning_rate = 0.0001
#learning_rate = 0.001
files_per_sample = 1
task_mode = BUILD_SEQUENCE
early_stopping = False
patience_epochs = 50

eval_run = 321
eval_epoch = 180



validation = True
microphones = []
# specific to data set
culture_limitations = ["B"]
age_limit = None
#time_steps = 0.05 #s
time_steps = 1. #s
vad_system = "child_vad2"
#vad_system = "None"
feature_type = ""



train_data_path = ""
test_data_path = ""
devel_data_path = ""

# Partition files
train_data_path = "data_split_files/train_sessions.csv"
devel_data_path = "data_split_files/devel_sessions.csv"
test_data_path = "data_split_files/test_sessions.csv"


result_dir = "src/results/deenigma_ser/"
run_dir, current_run = new_run_directory(result_dir, mode)
make_directory(result_dir + run_dir)
checkpoint_folder = "src/models/trained/deenigma_ser/"
make_directory(checkpoint_folder)
checkpoint_folder += run_dir
make_directory(checkpoint_folder)


# feature directory
feature_dir = "chunked_" + vad_system + "_" + str(time_steps) + "_features/"
# label directory
label_dir = "chunked_" + vad_system + "_" + str(time_steps) + "_labels/"
# evaluation_batch_size = batch_size


if mode == TRAIN_AND_EVALUATE:
    train_data_generator = Universal_Generator_from_Partition_File(train_data_path, feature_dir, label_dir,
                                                                   batch_size=batch_size, data_limitations=culture_limitations, microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, start_label_column=start_label_column, shuffle=True, label_name=label_name, task_mode=task_mode, loading_mode=LOAD_ALL, end_column_label=end_label_column)
    if validation == True:
        devel_data_generator = Universal_Generator_from_Partition_File(devel_data_path, feature_dir, label_dir,
                                                                       batch_size=evaluation_batch_size, data_limitations=culture_limitations, microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, start_label_column=start_label_column, shuffle = False, label_name=label_name, task_mode=task_mode, loading_mode=LOAD_ALL, end_column_label=end_label_column)
    else:
        devel_data_generator = None

    test_data_generator = Universal_Generator_from_Partition_File(test_data_path, feature_dir, label_dir, batch_size=evaluation_batch_size, data_limitations=culture_limitations, microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, feature_type=feature_type, start_label_column=start_label_column, label_name=label_name, task_mode=task_mode, loading_mode=LOAD_BATCHES, end_column_label=end_label_column)

    model_summary_file = result_dir + run_dir + "model_summary.txt"
    hyperparameter_file = result_dir + run_dir + "hyperparameter_overview.txt"
    save_hyperparameters(hyperparameter_file, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, optimizer=optimizer, task=task_mode)

    if task_mode == BUILD_SEQUENCE:
        #model = get_ser_model_deenigma_many_to_many(train_data_generator.feature_shape, learning_rate=learning_rate)
        model = get_ser_model(train_data_generator.feature_shape, learning_rate=learning_rate)

    save_model_callback = Train_Call_Back(checkpoint_folder, model_save_interval=model_callback_interval)
    callbacks = [save_model_callback]
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=patience_epochs))
    print("Training model")
    l = len(train_data_generator)
    print("epochs: " + str(epochs))
    history = model.fit(train_data_generator, validation_data=devel_data_generator, epochs=epochs, callbacks=callbacks)

    #X, Y = train_data_generator.__getitem__(0)
    #Y_pred = model.predict(X)

    history_path = result_dir + run_dir + "history.csv"
    history_df = pd.DataFrame(history.history)
    val_losses = history_df["val_loss"].values
    min_epoch = np.argmin(val_losses)
    print("Minimum Epoch: {}".format(min_epoch))
    with open(result_dir + run_dir + "min_epoch.txt", "w") as f:
        f.write("Minimum Epoch: {}".format(min_epoch))
    history_df.to_csv(history_path)
    print("Evaluating train results")
    model.evaluate(train_data_generator)
    print("Evaluating devel results")
    model.evaluate(devel_data_generator)
    print("Evaluating test results")
    model.evaluate(test_data_generator)

elif mode == EVALUATE:

    # test_data_generator = Universal_Generator_from_Partition_File(train_data_path, feature_dir, label_dir,
    #                                                                 batch_size=batch_size,
    #                                                                 data_limitations=culture_limitations, microphones=microphones,
    #                                                                 hop_size=hop_size, files_per_sample=files_per_sample,
    #                                                                 skip_label_columns=1, shuffle=False, task_mode=task_mode)
    train_data_generator = Universal_Generator_from_Partition_File(train_data_path, feature_dir, label_dir,
                                                                   batch_size=evaluation_batch_size,
                                                                   data_limitations=culture_limitations,
                                                                   microphones=microphones, hop_size=hop_size,
                                                                   files_per_sample=files_per_sample,
                                                                   start_label_column=start_label_column, shuffle=False,
                                                                   label_name=label_name, task_mode=task_mode,
                                                                   loading_mode=LOAD_BATCHES,
                                                                   end_column_label=end_label_column)
    devel_data_generator = Universal_Generator_from_Partition_File(devel_data_path, feature_dir, label_dir,
                                                                   batch_size=evaluation_batch_size,
                                                                   data_limitations=culture_limitations,
                                                                   microphones=microphones, hop_size=hop_size,
                                                                   files_per_sample=files_per_sample,
                                                                   start_label_column=start_label_column, shuffle=False,
                                                                   label_name=label_name, task_mode=task_mode,
                                                                   loading_mode=LOAD_BATCHES,
                                                                   end_column_label=end_label_column)
    test_data_generator = Universal_Generator_from_Partition_File(test_data_path, feature_dir, label_dir,
                                                                   batch_size=evaluation_batch_size,
                                                                   data_limitations=culture_limitations,
                                                                   microphones=microphones, hop_size=hop_size,
                                                                   files_per_sample=files_per_sample,
                                                                   start_label_column=start_label_column, shuffle=False,
                                                                   label_name=label_name, task_mode=task_mode,
                                                                   loading_mode=LOAD_BATCHES,
                                                                   end_column_label=end_label_column)

    #test_data_generator = Universal_Generator_from_Partition_File(test_data_path, feature_dir, label_dir, batch_size=evaluation_batch_size, data_limitations=culture_limitations, microphones=microphones, hop_size=hop_size, files_per_sample=files_per_sample, feature_type=feature_type, skip_label_columns=1, label_name=label_name, task_mode=task_mode)
    model_path = "src/models/trained/deenigma_ser/run_" + str(eval_run).zfill(3) + "/model_" + str(eval_epoch).zfill(3) + ".h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=[ccc_tf, "mse"])
    save_dir = result_dir + run_dir + "images/"
    make_directory(save_dir)
    print("---------------------------------------------------------------------------")
    print("Train")
    #save_history(model.evaluate(train_data_generator), result_dir + run_dir, partition="Train")
    #evaluate_data_generator(train_data_generator, model, result_dir + run_dir, partition="Train")
    #plot_data_generator_predictions(train_data_generator, model, save_dir, "train")
    print("---------------------------------------------------------------------------")
    print("Devel")
    #save_history(model.evaluate(devel_data_generator), result_dir + run_dir, partition="Devel")
    evaluate_data_generator(devel_data_generator, model, result_dir + run_dir, partition="Devel")
    #plot_data_generator_predictions(devel_data_generator, model, save_dir, "devel")
    print("---------------------------------------------------------------------------")
    print("Test")
    #save_history(model.evaluate(test_data_generator), result_dir + run_dir, partition="Test")
    evaluate_data_generator(test_data_generator, model, result_dir + run_dir, partition="Test")
    #plot_data_generator_predictions(test_data_generator, model, save_dir, "test")


