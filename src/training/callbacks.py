import tensorflow as tf

class Train_Call_Back(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_folder, model_save_interval=1, result_dir=None):
        self.checkpoint_folder = checkpoint_folder
        self.model_save_interval = model_save_interval
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.model_save_interval == 0:
        #print("saving: " + self.checkpoint_folder + "model_" + str(epoch).zfill(3) + ".h5")
            self.model.save(self.checkpoint_folder + "model_" + str(epoch).zfill(3) + ".h5")

