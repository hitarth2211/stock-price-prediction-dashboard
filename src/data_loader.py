import pandas as pd

def load_data(file_path):
    """
    Load stock data from CSV.
    """
    df = pd.read_csv(file_path)

    # Remove unwanted rows
    df = df.iloc[2:].copy()

    # Rename columns
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Convert data types
    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # Set Date as index
    df.set_index("Date", inplace=True)

    return df