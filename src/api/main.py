from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# Chemin robuste vers le modèle
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "xgb_fraud.pkl")
model = joblib.load(model_path)

app = FastAPI(
    title="Détection de Fraude Bancaire - MLOps 2025",
    description="API XGBoost + DVC + MLflow + Evidently",
    version="1.0.0"
)

class Transaction(BaseModel):
    features: list[float] = Field(..., example=[0, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098, 0.364, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468, -0.470, 0.208, 0.025, 0.404, 0.251, -0.018, 0.277, 0.326, -0.189, 0.003, -0.204, -0.021, 0.059, 150])

@app.get("/")
def home():
    return {"message": "API Fraude en ligne !", "docs": "/docs"}

@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.features])
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "probability_fraud": round(proba, 5),
        "is_fraud": proba > 0.5
    }
    