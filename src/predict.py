import joblib
import pandas as pd


def load_model(model_path):
    """
    Load the trained model.
    """
    return joblib.load(model_path)


def predict_next_day(model, features):
    """
    Predict the next day's closing price.

    Parameters:
        model : Trained ML model
        features : DataFrame containing one row of features

    Returns:
        Predicted price
    """
    prediction = model.predict(features)

    return prediction[0]