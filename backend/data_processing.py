import os
import pandas as pd
import ta

def process_msft_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_path = os.path.join(data_dir, "msft_raw.csv")
    output_path = os.path.join(data_dir, "msft_processed.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError("Run data_collection.py first")

    df = pd.read_csv(input_path)

    # Sort by date
    df = df.sort_values("Date")

    # =========================
    # FEATURE ENGINEERING
    # =========================

    # Moving averages
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()

    # Returns & momentum
    df["Return"] = df["Close"].pct_change()
    df["Momentum"] = df["Close"].pct_change(5)

    # Volatility
    df["Volatility"] = df["Close"].rolling(10).std()

    # =========================
    # TARGET VARIABLE
    # =========================

    df["Next_Close"] = df["Close"].shift(-1)

    # 1 = UP, 0 = DOWN
    df["Trend"] = (df["Next_Close"] > df["Close"]).astype(int)

    # =========================
    # CLEAN DATA
    # =========================

    df = df.dropna()

    # Save processed data
    df.to_csv(output_path, index=False)

    print("Processed dataset saved!")
    print("Rows:", len(df))
    print(df.head())

if __name__ == "__main__":
    process_msft_data()
