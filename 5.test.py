import os
import warnings
warnings.filterwarnings("ignore")

import sys, os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dot, BatchNormalization
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping

import gc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from keras.utils import Sequence

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")
    exit()

sys.path.insert(1, os.path.join(sys.path[0], ".."))
idx = pd.IndexSlice

results_path = Path("KR2_results", "asset_pricing")
if not results_path.exists():
    results_path.mkdir(parents=True)

np.random.seed(42)

class DataGenerator(Sequence):
    def __init__(self, X1, X2, y, batch_size):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X1) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X1 = self.X1[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_X2 = self.X2[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return [batch_X1, batch_X2], batch_y


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

characteristics = [
    "beta",
    "betasq",
    "chmom",
    "krwvol",
    "idiovol",
    "ill",
    "indmom",
    "maxret",
    "mom12m",
    "mom1m",
    "mom36m",
    "mvel",
    "retvol",
    "turn",
    "turn_std",
]

data = pd.read_hdf(results_path / "autoencoder.h5", "model_data")
n_factors = 3
n_characteristics = len(characteristics)
n_tickers = len(data.index.unique("ticker"))

def make_model(hidden_units=8, n_factors=3):
    input_beta = Input((n_tickers, n_characteristics), name="input_beta")
    input_factor = Input((n_tickers,), name="input_factor")

    hidden_layer = Dense(units=hidden_units, activation="relu", name="hidden_layer")(
        input_beta
    )
    batch_norm = BatchNormalization(name="batch_norm")(hidden_layer)

    output_beta = Dense(units=n_factors, name="output_beta")(batch_norm)

    pfo_factor = Dense(units=n_characteristics, name="pfo_factor")(input_factor)
    output_factor = Dense(units=n_factors, name="output_factor")(pfo_factor)

    output = Dot(axes=(2, 1), name="output_layer")([output_beta, output_factor])

    model = Model(inputs=[input_beta, input_factor], outputs=output)
    model.compile(loss="mse", optimizer="adam")
    return model

def get_train_valid_data(data, train_idx, val_idx):
    train, val = data.iloc[train_idx], data.iloc[val_idx]
    X1_train = train.loc[:, characteristics].values.reshape(
        -1, n_tickers, n_characteristics
    )
    X1_val = val.loc[:, characteristics].values.reshape(
        -1, n_tickers, n_characteristics
    )
    X2_train = train.loc[:, "returns"].unstack("ticker")
    X2_val = val.loc[:, "returns"].unstack("ticker")
    y_train = train.returns_fwd.unstack("ticker")
    y_val = val.returns_fwd.unstack("ticker")
    return X1_train, X2_train, y_train, X1_val, X2_val, y_val

# define early stopping callback
early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min")

from utils import MultipleTimeSeriesCV, format_time
YEAR = 52
cv = MultipleTimeSeriesCV(
    n_splits = 5,  # 5
    train_period_length = 17 * YEAR,
    test_period_length = 1 * YEAR,
    lookahead = 1,
)

########################################################################
# 대충 좋은 에포크 입력하는 코드
n_factors = 2
units = 8
batch_size = 256
first_epoch = 220
last_epoch = 250

predictions = []
for epoch in tqdm(list(range(first_epoch, last_epoch))):
    epoch_preds = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(data)):
        X1_train, X2_train, y_train, X1_val, X2_val, y_val = get_train_valid_data(
            data, train_idx, val_idx
        )

        train_gen = DataGenerator(X1_train, X2_train, y_train, batch_size=batch_size)
        val_gen = DataGenerator(X1_val, X2_val, y_val, batch_size=batch_size)
        model = make_model(hidden_units=units, n_factors=n_factors)
        model.fit_generator(
            train_gen,
            validation_data=val_gen,
            epochs=epoch,
            verbose=0,
            shuffle=True,
            callbacks=[ClearMemory(), early_stop],
        )
        epoch_preds.append(
            pd.Series(
                model.predict_generator(
                    val_gen, callbacks=[ClearMemory(), early_stop]
                ).reshape(-1),
                index=y_val.stack().index,
            ).to_frame(epoch)
        )

    predictions.append(pd.concat(epoch_preds))

predictions_combined = pd.concat(predictions, axis=1).sort_index()
predictions_combined.to_hdf(results_path / "predictions.h5", "predictions")