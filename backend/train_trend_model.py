import os
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_trend_model():

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_path = os.path.join(data_dir, "msft_processed.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError("Run data_processing.py first")

    df = pd.read_csv(input_path)

    # =========================
    # FEATURES
    # =========================

    features = [
        "MA10",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "Return",
        "Momentum",
        "Volatility"
    ]

    X = df[features]
    y = df["Trend"]

    # =========================
    # TRAIN-TEST SPLIT
    # =========================

    # Important: no shuffle for time series
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # =========================
    # MODEL
    # =========================

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =========================
    # EVALUATION
    # =========================

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print("\nModel Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    # =========================
    # SAVE MODEL
    # =========================

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "trend_model.pkl")

    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path}")


if __name__ == "__main__":
    train_trend_model()
