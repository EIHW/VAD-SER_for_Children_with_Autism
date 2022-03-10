from src.utils.csv_utils import save_batch
from src.utils.path_utils import make_directory
import pandas as pd
from contextlib import redirect_stdout

def save_model_summary(model, path):
    with open(path, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def save_hyperparameters(path, optimizer="", learning_rate=0.0, epochs=0, batch_size=32, task=""):
    df = pd.DataFrame([["optimizer", optimizer], ["learning rate", learning_rate], ["epochs", epochs], ["batch_size", batch_size], ["task", task]], columns=["hyperparameter", "value"])
    df.to_csv(path, index=False)


def infere_write_from_data_generator(datagenerator, model, out_dir):
    make_directory(out_dir)
    for i in range(len(datagenerator)):
        batch_indices = datagenerator.indices[i * datagenerator.batch_size:(i + 1) * datagenerator.batch_size]
        feature_files = datagenerator.feature_files[batch_indices]
        print(feature_files)
        X, Y = datagenerator.__getitem__(i)
        predictions = model.predict(X)
        save_batch(predictions, feature_files, out_dir)
        print()
