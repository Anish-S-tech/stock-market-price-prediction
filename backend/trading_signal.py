import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import ta

from backend.news_fetcher import NewsFetcher
from backend.sentiment_model import SentimentModel


class TradingSignalEngine:

    def __init__(self):

        base_dir = os.path.join(os.path.dirname(__file__), "..")

        # Load models safely
        self.trend_model = joblib.load(os.path.join(base_dir, "models", "trend_model.pkl"))
        self.price_model = tf.keras.models.load_model(os.path.join(base_dir, "models", "price_model.h5"), compile=False)
        self.price_model.compile(optimizer="adam", loss="mse")
        self.scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))

        self.news = NewsFetcher()
        self.sentiment = SentimentModel()

    def fetch_data(self):
        df = yf.download("MSFT", period="6mo")
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def compute_features(self, df):

        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        # RSI
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()

        df["Return"] = df["Close"].pct_change()
        df["Momentum"] = df["Close"].pct_change(5)
        df["Volatility"] = df["Close"].rolling(10).std()

        return df.dropna()

    def predict_trend(self, df):

        features = df.iloc[-1][[
            "MA10", "MA20", "MA50", "RSI", "MACD",
            "Return", "Momentum", "Volatility"
        ]].values.reshape(1, -1)

        return self.trend_model.predict(features)[0]

    def predict_price(self, df):

        prices = df["Close"].values.reshape(-1, 1)

        scaled = self.scaler.transform(prices)

        seq = scaled[-30:].reshape(1, 30, 1)

        pred = self.price_model.predict(seq, verbose=0)

        return float(self.scaler.inverse_transform(pred)[0][0])

    def get_sentiment(self):

        headlines = self.news.get_headlines("MSFT")

        result = self.sentiment.overall_sentiment(headlines)

        return result["overall_sentiment"]

    def generate_signal(self):

        df = self.fetch_data()
        df = self.compute_features(df)

        current_price = df["Close"].iloc[-1]

        trend = self.predict_trend(df)
        predicted_price = self.predict_price(df)
        sentiment = self.get_sentiment()

        # =========================
        # DECISION LOGIC
        # =========================

        if trend == 1 and predicted_price > current_price and sentiment == "Positive":
            signal = "BUY"

        elif trend == 0 and sentiment == "Negative":
            signal = "SELL"

        else:
            signal = "HOLD"

        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "trend": "UP" if trend == 1 else "DOWN",
            "sentiment": sentiment,
            "signal": signal
        }
