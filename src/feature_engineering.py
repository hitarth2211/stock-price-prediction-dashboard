import pandas as pd


def create_features(df):
    """
    Create features for stock price prediction.
    """

    # Moving averages
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # Volatility
    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    # Lag features
    df["Close_1"] = df["Close"].shift(1)
    df["Close_2"] = df["Close"].shift(2)
    df["Close_3"] = df["Close"].shift(3)
    df["Close_5"] = df["Close"].shift(5)

    return df