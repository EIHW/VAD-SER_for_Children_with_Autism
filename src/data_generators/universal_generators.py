import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from src.utils.partition_file_utils import get_audio_files_and_class_labels_from_csv, get_feature_label_files_deenigma
from src.utils.csv_utils import get_features_from_file, get_delimeter, load_and_process_feature_single_file, load_and_process_features_build_sequence
from src.utils.constants import *
from pickle import load

class Universal_Generator_from_Partition_File(Sequence):
    def __init__(self, partition_file="", feature_dir="", label_dir="", batch_size=32, shuffle=True, first_feature_column=0, last_feature_column=-1, only_one_batch=False, data_limitations = None, database=None, start_label_column=2, files_per_sample=1, hop_size=1, microphones=[4], feature_type ="", normaliser_path = None, inference=False, label_name="", task_mode="many_to_many", end_column_label=-1, loading_mode = LOAD_BATCHES):
        print("Initialising dataloader for: " + str(partition_file))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partition_file = partition_file
        self.label_dir = label_dir
        self.loading_mode = loading_mode
        self.microphones = microphones
        self.data_limitations = data_limitations
        self.database = database
        self.files_per_sample = files_per_sample
        self.hop_size = hop_size
        self.label_name = label_name
        self.inference = inference
        self.task_mode = task_mode
        self.feature_dir = feature_dir
        self.scaler = None
        if normaliser_path != None:
            self.scaler = load(open(normaliser_path, 'rb'))
        self.feature_files, self.label_files = get_feature_label_files_deenigma(partition_file, feature_dir=feature_dir, feature_type=feature_type, label_dir=label_dir, only_one_batch=only_one_batch, data_limitations=data_limitations, database=database, inference=inference, microphones=self.microphones, task_mode=task_mode)
        if hop_size == 1 and files_per_sample == 1:
            self.indices = np.arange(len(self.feature_files))
        else:
            self.indices = np.arange(self.calculate_n_samples() - 1)
        self.on_epoch_end()
        if len(self.feature_files) == 0:
            print("Feature files cannot be loaded!!!")
        if self.task_mode == BUILD_SEQUENCE:
            self.feature_file_delimiter = get_delimeter(self.feature_files[0][0])
        else:
            self.feature_file_delimiter = get_delimeter(self.feature_files[0])


        self.start_column_feature_file = first_feature_column
        self.end_column_feature_file = last_feature_column

        if not inference:
            if self.task_mode == BUILD_SEQUENCE:
                self.label_file_delimiter = get_delimeter(self.label_files[0][0])
            else:
                self.label_file_delimiter = get_delimeter(self.label_files[0])
            # TODO: different for
            self.start_column_label = start_label_column
            self.end_column_label = end_column_label
        if hop_size == 1 and files_per_sample == 1:
            if self.task_mode == BUILD_SEQUENCE:
                self.feature_shape = get_features_from_file(self.feature_files[0][0],
                                                            start_column=self.start_column_feature_file,
                                                            end_column=self.end_column_feature_file,
                                                            delimiter=self.feature_file_delimiter).shape
            else:
                self.feature_shape = get_features_from_file(self.feature_files[0], start_column=self.start_column_feature_file, end_column=self.end_column_feature_file, delimiter=self.feature_file_delimiter).shape
        else:
            X, Y = self.__getitem__(0)
            self.feature_shape = X.shape[1:]
        if self.loading_mode == LOAD_ALL:
            self.X, self.Y = self.__load_all_data()
        else:
            self.X = None
            self.Y = None
        print("Done!")


    def calculate_n_samples(self):
        n_files = self.feature_files.shape[0]
        n_samples = n_files + 1 - self.files_per_sample
        n_samples = int(np.ceil(n_samples / self.hop_size))
        return n_samples

    def __len__(self):
        """
        Calculates number of batches in one epoch
        :return: number of batches per epoch
        """
        if self.loading_mode == LOAD_BATCHES:
            n_files = len(self.feature_files)
            n_samples = n_files + 1 - self.files_per_sample
            n_samples = int(np.ceil(n_samples/self.hop_size))
            n_batches = int(np.ceil(n_samples/ self.batch_size))
            return n_batches
        else:
            return int(np.ceil((self.X.shape[0])/self.batch_size))
            # return int(self.X.shape[0])

    def on_epoch_end(self):
        """
        Shuffles the order of examples for next epoch
        :return: None
        """
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Determines X and Y for current batch of data
        :param index: index of current of batch [0;__len__]
        :return: X and Y as np.arrays
        """
        #TODO: the code only works for all microphones rn
        self.batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        if self.loading_mode == LOAD_BATCHES:
            if self.files_per_sample == 1 and self.hop_size == 1:
                if self.loading_mode == LOAD_BATCHES and self.task_mode == BUILD_SEQUENCE:
                    batch_feature_files = []
                    for i in self.batch_indices:
                        batch_feature_files.append(self.feature_files[i])
                else:
                    batch_feature_files = self.feature_files[self.batch_indices]
                if len(self.label_files) == 0:
                    batch_label_files = np.array([])
                else:
                    if self.loading_mode == LOAD_BATCHES and self.task_mode == BUILD_SEQUENCE:
                        batch_label_files = []
                        for i in self.batch_indices:
                            batch_label_files.append(self.label_files[i])
                    else:
                        batch_label_files = self.label_files[self.batch_indices]
            else:
                # this is supposed to generate a batch with samples consisting of multiple files, i.e., an np array of size (batch,size, files_per_sample)
                # wrong lexicographic order
                # batch_indices_adapted = (self.batch_indices // n_microphones) * self.hop_size * n_microphones + self.batch_indices % n_microphones
                batch_indices_adapted = self.batch_indices * self.hop_size
                batch_index_matrix = batch_indices_adapted[:,np.newaxis]
                # file convention for de-enigma data is:
                files_per_sample_array = np.arange(0, self.files_per_sample)
                batch_index_matrix = batch_index_matrix + files_per_sample_array
                batch_feature_files = self.feature_files[batch_index_matrix]
                if len(self.label_files) == 0:
                    batch_label_files = np.array([])
                else:
                    batch_label_files = self.label_files[batch_index_matrix]
            X_batch, Y_batch = self.__data_generation(batch_feature_files, batch_label_files)

        elif self.loading_mode == LOAD_ALL:
            X_batch = self.X[self.batch_indices]
            Y_batch = self.Y[self.batch_indices]

        return X_batch, Y_batch

    def __load_all_data(self):
        return self.__data_generation(self.feature_files, self.label_files)

    def __data_generation(self, batch_feature_files, batch_label_files):
        """
        Loads data for a given batch and converts it to spectrograms.
        :param batch_feature_files: np.array of files containing features of current batch
        :param batch_labels: labels for currnt batch
        :return: X, Y as np.arrays
        """
        # TODO: For now only sequential data
        # TODO: unify cases with multiple files per sample and hop sizes
        if self.files_per_sample == 1 and self.hop_size == 1:
            features = []
            if self.task_mode == BUILD_SEQUENCE:
                max_len = 0
                for batch_feature_file_set in batch_feature_files:
                    new_features, feat_len = load_and_process_features_build_sequence(batch_feature_file_set,
                                                                                  start_column=self.start_column_feature_file,
                                                                                  delimiter=self.feature_file_delimiter,
                                                                                  end_column=self.end_column_feature_file,
                                                                                  scaler=self.scaler)
                    max_len = max(max_len, feat_len)
                    features.append(new_features)
                    # TODO: Quick test
                    # break
            else:
                max_len = 0
                for batch_feature_file in batch_feature_files:
                    new_features, feat_len = load_and_process_feature_single_file(batch_feature_file,
                                                         start_column=self.start_column_feature_file,
                                                         delimiter=self.feature_file_delimiter,
                                                         end_column=self.end_column_feature_file, scaler=self.scaler)
                    max_len = max(max_len, feat_len)
                    features.append(new_features)
                    # TODO: Quick test
                    # break

            # zero padding
            labels = []
            if self.inference:
                Y = np.array([])
            else:
                if self.task_mode == BUILD_SEQUENCE:
                    max_len = 0
                    for batch_label_file_set in batch_label_files:
                        new_labels, lab_len = load_and_process_features_build_sequence(batch_label_file_set,
                                                                                          start_column=self.start_column_label,
                                                                                          delimiter=self.label_file_delimiter,
                                                                                          end_column=self.end_column_label,
                                                                                          scaler=self.scaler)
                        max_len = max(max_len, lab_len)
                        labels.append(new_labels)
                else:
                    for batch_label_file in batch_label_files:
                        new_labels, lab_len = load_and_process_feature_single_file(batch_label_file,
                                                                                   start_column=self.start_column_label,
                                                                                   delimiter=self.label_file_delimiter,
                                                                                   end_column=self.end_column_label)
                        # if self.label_name == "arousal":
                        #     lab = lab[:,:,:1]
                        # elif self.label_name == "valence":
                        #     lab = lab[:, :, 1:]
                        labels.append(new_labels)
                        max_len = max(max_len, lab_len)
                for i in range(len(labels)):
                    labels[i] = np.concatenate(
                        (labels[i], np.zeros((1, max_len - labels[i].shape[1], labels[i].shape[2]))), axis=1)
                Y = np.vstack(labels)
            for i in range(len(features)):
                features[i] = np.concatenate((features[i], np.zeros((1, max_len - features[i].shape[1], features[i].shape[2]))), axis=1)

            X = np.vstack(features)
        else:
            features = []
            for batch_feature_file_rows in batch_feature_files:
                feature_row = []
                for batch_feature_file in batch_feature_file_rows:
                    feature_row.append(get_features_from_file(batch_feature_file, start_column=self.start_column_feature_file,
                                                              delimiter=self.feature_file_delimiter))
                feature_row = np.vstack(feature_row)[np.newaxis, ...]
                features.append(feature_row)
            X = np.vstack(features)

            labels = []
            for batch_label_file_rows in batch_label_files:
                label_rows = []
                for batch_label_file in batch_label_file_rows:
                    label_rows.append(get_features_from_file(batch_label_file, start_column=self.start_column_label,
                                                             delimiter=self.label_file_delimiter))
                label_rows = np.vstack(label_rows)[np.newaxis, ...]
                labels.append(label_rows)
            if self.inference:
                Y = np.array([])
            else:
                Y = np.vstack(labels)
        #return np.zeros((100,430,100)), np.zeros((100,430,1))
        #print("X: {}, Y: {}".format(X.shape, Y.shape))
        if self.task_mode == MANY_TO_ONE:
            Y = Y[:, -1, :]
        elif self.task_mode == CNN_TO_ONE:
            Y = Y[:, -1, :]
            X = X[..., np.newaxis]
        elif self.task_mode == ONE_TO_ONE:
            Y = Y[:, 0, :]
            X = X[:, 0, :]
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)
        return X, Y


class deenigma_ser_generator(Sequence):
    def __init__(self, partition_file, feature_dir, label_dir, batch_size=32, shuffle=True, only_one_batch=False, data_limitations = None, database=None, mode=TRAIN_AND_EVALUATE, files_per_sample=1, hop_size=1, microphones=4, feature_type = "", normaliser_path = None, inference=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partition_file = partition_file
        self.label_dir = label_dir
        self.microphones = microphones
        self.mode = mode
        self.data_limitations = data_limitations
        self.database = database
        self.files_per_sample = files_per_sample
        self.hop_size = hop_size
        self.inference = inference
        self.feature_dir = feature_dir
        self.scaler = None
        if normaliser_path != None:
            self.scaler = load(open(normaliser_path, 'rb'))
        self.feature_files, self.label_files = get_feature_label_files_deenigma(partition_file, feature_dir=feature_dir, feature_type=feature_type, label_dir=label_dir, only_one_batch=only_one_batch, data_limitations=data_limitations, database=database, inference=inference)
        if hop_size == 1 and files_per_sample == 1:
            self.indices = np.arange(len(self.feature_files))
        else:
            self.indices = np.arange(self.calculate_n_samples() - 1)
        self.on_epoch_end()
        if len(self.feature_files) == 0:
            print("Feature files cannot be loaded!!!")
        self.feature_file_delimiter = get_delimeter(self.feature_files[0])
        self.skip_columns_feature_file = 2
        if not inference:
            self.label_file_delimiter = get_delimeter(self.label_files[0])
            self.skip_columns_label_file = 2
        if hop_size == 1 and files_per_sample == 1:
            self.feature_shape = get_features_from_file(self.feature_files[0], start_column=self.skip_columns_feature_file, delimiter=self.feature_file_delimiter).shape
        else:
            X, Y = self.__getitem__(0)
            self.feature_shape = X.shape[1:]
    def __getitem__(self, index):
        """
        Determines X and Y for current batch of data
        :param index: index of current of batch [0;__len__]
        :return: X and Y as np.arrays
        """
        #TODO: the code only works for all microphones rn
        n_microphones = self.microphones
        self.batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        if self.files_per_sample == 1 and self.hop_size == 1:
            batch_feature_files = self.feature_files[self.batch_indices]
            if len(self.label_files) == 0:
                batch_label_files = np.array([])
            else:
                batch_label_files = self.label_files[self.batch_indices]
        else:
            # this is supposed to generate a batch with samples consisting of multiple files, i.e., an np array of size (batch,size, files_per_sample)
            # wrong lexicographic order
            # batch_indices_adapted = (self.batch_indices // n_microphones) * self.hop_size * n_microphones + self.batch_indices % n_microphones
            batch_indices_adapted = self.batch_indices * self.hop_size
            batch_index_matrix = batch_indices_adapted[:,np.newaxis]
            # file convention for de-enigma data is:
            files_per_sample_array = np.arange(0, self.files_per_sample)
            batch_index_matrix = batch_index_matrix + files_per_sample_array
            batch_feature_files = self.feature_files[batch_index_matrix]
            if len(self.label_files) == 0:
                batch_label_files = np.array([])
            else:
                batch_label_files = self.label_files[batch_index_matrix]
        return self.__data_generation(batch_feature_files, batch_label_files)

    def __data_generation(self, batch_feature_files, batch_label_files):
        """
        Loads data for a given batch and converts it to spectrograms.
        :param batch_feature_files: np.array of files containing features of current batch
        :param batch_labels: labels for currnt batch
        :return: X, Y as np.arrays
        """
        # TODO: For now only sequential data
        # TODO: unify cases with multiple files per sample and hop sizes
        if self.files_per_sample == 1 and self.hop_size == 1:
            features = []
            for batch_feature_file in batch_feature_files:
                item_features = get_features_from_file(batch_feature_file, start_column=self.skip_columns_feature_file,
                                                       delimiter=self.feature_file_delimiter)
                if self.scaler != None:
                    item_features = self.scaler.fit_transform(item_features)
                item_features = item_features[np.newaxis, ...]
                features.append(item_features)
            X = np.vstack(features)
            labels = []

            if self.inference:
                Y = np.array([])
            else:
                for batch_label_file in batch_label_files:
                    labels.append(get_features_from_file(batch_label_file, start_column=self.skip_columns_label_file,
                                                         delimiter=self.label_file_delimiter)[np.newaxis, ...])
                Y = np.vstack(labels)
        else:
            features = []
            for batch_feature_file_rows in batch_feature_files:
                feature_row = []
                for batch_feature_file in batch_feature_file_rows:
                    feature_row.append(get_features_from_file(batch_feature_file, start_column=self.skip_columns_feature_file,
                                                              delimiter=self.feature_file_delimiter))
                feature_row = np.vstack(feature_row)[np.newaxis, ...]
                features.append(feature_row)
            X = np.vstack(features)

            labels = []
            for batch_label_file_rows in batch_label_files:
                label_rows = []
                for batch_label_file in batch_label_file_rows:
                    label_rows.append(get_features_from_file(batch_label_file, start_column=self.skip_columns_label_file,
                                                             delimiter=self.label_file_delimiter))
                label_rows = np.vstack(label_rows)[np.newaxis, ...]
                labels.append(label_rows)
            if self.inference:
                Y = np.array([])
            else:
                Y = np.vstack(labels)
        return X, Y





