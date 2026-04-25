import joblib
import pandas as pd
from config import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict_traffic(model, hour, day, lag1, lag2):
    data = pd.DataFrame([{
        'hour': hour,
        'day': day,
        'lag1': lag1,
        'lag2': lag2
    }])

    prediction = model.predict(data)[0]

    return prediction


def classify_congestion(value):
    if value < 20:
        return "Low"
    elif value < 50:
        return "Medium"
    else:
        return "High"