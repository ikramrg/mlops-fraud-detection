import joblib
import pandas as pd
import os

def test_model_exists():
    model_path = "models/xgb_fraud.pkl"
    assert os.path.exists(model_path)

def test_prediction_output():
    model = joblib.load("models/xgb_fraud.pkl")
    sample = pd.DataFrame([{
        col: 0.0 for col in range(30)
    }])  # 30 features
    proba = model.predict_proba(sample)[0, 1]
    assert 0 <= proba <= 1