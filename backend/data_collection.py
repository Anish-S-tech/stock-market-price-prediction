import os
import pandas as pd
import yfinance as yf

def collect_msft_data(start="2010-01-01", end=None, interval="1d"):
    """
    Download MSFT historical data and save to CSV.
    """
    ticker = "MSFT"
    print(f"Downloading data for {ticker}...")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True
    )

    if df.empty:
        raise ValueError("No data downloaded. Check internet or ticker.")

    # Reset index so Date becomes a column
    df.reset_index(inplace=True)

    # Ensure consistent column names
    df.columns = [col[0] if isinstance(col, tuple) else str(col).replace(" ", "_") for col in df.columns]

    # Create data directory if not exists
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "msft_raw.csv")
    df.to_csv(file_path, index=False)

    print(f"Saved dataset to {file_path}")
    print(f"Rows: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    collect_msft_data()
