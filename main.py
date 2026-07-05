from src.data_loader import load_data
from src.feature_engineering import create_features
from src.train import train_model

# Load data
df = load_data("data/raw/AAPL.csv")

# Create features
df = create_features(df)

# Train model
model = train_model(df)