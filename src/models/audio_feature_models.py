from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, Bidirectional, SimpleRNN, MultiHeadAttention, Input, Masking, TimeDistributed, Conv2D, Activation, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from src.evaluation.losses import ccc_loss, ccc, ccc_loss_multiple_targets
from src.evaluation.evaluation import ccc_tf
from src.utils.model_utils import save_model_summary



def get_ser_model(input_shape, learning_rate = 0.01):
    adam = Adam(learning_rate=learning_rate)
    x_input = Input((None, input_shape[-1]))
    x = Masking()(x_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out1 = Dense(1)(x)
    model = Model(inputs=x_input, outputs=out1)
    model.compile(optimizer=adam, loss="mse", metrics=[ccc_tf, "mse", RootMeanSquaredError(name='rmse', dtype=None)])
    model.summary()
    return model


def get_detection_model(input_shape, learning_rate=0.01):
    adam = Adam(learning_rate=learning_rate)
    x_input = Input((None, input_shape[-1]))
    y_input = Input((None, 1))
    x = Dense(128, activation='relu')(x_input)
    x = Bidirectional(LSTM(128, return_sequences=True))(x_input)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = MultiHeadAttention(num_heads=1, key_dim=128)(x, x, x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)
    model.compile(optimizer=adam, loss='mse')
    model.summary()
    return model



