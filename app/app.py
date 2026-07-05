import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Project Path Setup
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import load_model, predict_next_day

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📊 Dashboard")

st.sidebar.markdown("---")

stock = st.sidebar.selectbox(
    "Choose a Stock",
    ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Machine Learning Model")
st.sidebar.success("Linear Regression")

# ----------------------------
# Company Names
# ----------------------------
company_names = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc."
}

# ----------------------------
# Title
# ----------------------------
st.title("📈 Stock Price Prediction Dashboard")

st.markdown(
    """
Welcome! This dashboard predicts the **next trading day's closing price**
using a **Machine Learning Linear Regression model**.

Select a stock from the sidebar to explore historical prices,
technical indicators and tomorrow's predicted closing price.
"""
)

st.success(f"Currently Viewing **{company_names[stock]} ({stock})**")

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(
    BASE_DIR / "data" / "processed" / f"{stock}_processed.csv",
    index_col=0,
    parse_dates=True
)

df.reset_index(inplace=True)
df.rename(columns={"index": "Date"}, inplace=True)

# ----------------------------
# Stock Overview
# ----------------------------
st.subheader("📌 Stock Overview")

latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest Close",
        f"${latest['Close']:.2f}"
    )

with col2:
    st.metric(
        "Volume",
        f"{int(latest['Volume']):,}"
    )

with col3:
    st.metric(
        "20-Day Average",
        f"${latest['MA_20']:.2f}"
    )

with col4:
    st.metric(
        "50-Day Average",
        f"${latest['MA_50']:.2f}"
    )

# ----------------------------
# Historical Data
# ----------------------------
with st.expander("📄 View Historical Data"):
    st.dataframe(df.tail(20), use_container_width=True)

# ----------------------------
# Plotly Chart
# ----------------------------
st.subheader("📊 Stock Price & Moving Averages")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Closing Price"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA_20"],
        mode="lines",
        name="20-Day Average"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA_50"],
        mode="lines",
        name="50-Day Average"
    )
)

fig.update_layout(
    title=f"{stock} Stock Price History",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template="plotly_white",
    height=550,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Moving Average Explanation
# ----------------------------
st.info(
    """
### 📚 Understanding the Chart

**Closing Price**
- Actual market closing price for each trading day.

**20-Day Moving Average**
- Average closing price over the last **20 trading days**.
- Shows the short-term market trend.

**50-Day Moving Average**
- Average closing price over the last **50 trading days**.
- Shows the long-term market trend.

Investors often compare these averages to identify trend changes.
"""
)

# ----------------------------
# Load Model & Metrics
# ----------------------------
model = load_model(
    BASE_DIR / "models" / f"{stock}_model.pkl"
)

with open(BASE_DIR / "models" / f"{stock}_metrics.json", "r") as f:
    metrics = json.load(f)

# ----------------------------
# Prediction
# ----------------------------
latest_features = df.tail(1)[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_5",
        "MA_10",
        "MA_20",
        "MA_50",
        "Daily_Return",
        "Volatility",
        "Close_1",
        "Close_2",
        "Close_3",
        "Close_5",
    ]
]

prediction = predict_next_day(model, latest_features)

current_price = latest["Close"]

change = prediction - current_price
percent_change = (change / current_price) * 100

# ----------------------------
# Prediction Summary
# ----------------------------
st.subheader("🔮 Tomorrow's Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Current Closing Price",
        f"${current_price:.2f}"
    )

with col2:
    st.metric(
        "Predicted Closing Price",
        f"${prediction:.2f}",
        delta=f"{change:.2f} ({percent_change:.2f}%)"
    )

# ----------------------------
# Model Performance
# ----------------------------
st.subheader("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", metrics["MAE"])

with col2:
    st.metric("RMSE", metrics["RMSE"])

with col3:
    st.metric("R² Score", metrics["R2"])

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.caption(
    "Developed by Hitarth Shah | "
    "Machine Learning • Streamlit • Scikit-learn • Plotly"
)