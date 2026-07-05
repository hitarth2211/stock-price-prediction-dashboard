from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def train_model(df, model_name="linear_regression.pkl"):
    """
    Train a Linear Regression model.
    """

    # Create Target
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    # Features
    X = df[
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

    y = df["Target"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Model Performance")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Create models folder if it doesn't exist
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / model_name
    joblib.dump(model, model_path)

    print(f"Model saved at: {model_path}")

    print("Model saved successfully!")

    return model, mae, rmse, r2