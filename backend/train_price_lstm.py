import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def train_lstm():

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_path = os.path.join(data_dir, "msft_processed.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError("Run data_processing.py first")

    df = pd.read_csv(input_path)

    prices = df["Close"].values.reshape(-1, 1)

    # =========================
    # SCALE DATA
    # =========================
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # =========================
    # CREATE SEQUENCES
    # =========================
    seq_len = 30

    X, y = [], []

    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # =========================
    # MODEL
    # =========================
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # =========================
    # TRAIN
    # =========================
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # =========================
    # SAVE MODEL
    # =========================
    model.save(os.path.join(models_dir, "price_model.h5"))

    print("LSTM model saved!")
    print("Scaler saved!")


if __name__ == "__main__":
    train_lstm()
