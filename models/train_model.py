import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from utils.preprocessing import load_and_preprocess_data
from config import DATA_PATH, MODEL_PATH


def train_model():
    df = load_and_preprocess_data(DATA_PATH)

    features = ['hour', 'day', 'lag1', 'lag2']
    X = df[features]
    y = df['Vehicles']

    # Time-based split (important)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", mean_squared_error(y_test, predictions, squared=False))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
