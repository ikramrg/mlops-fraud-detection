from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
import joblib
import os

# Chemin du modèle
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "xgb_fraud.pkl"

# Chargement conditionnel : 
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    print(f"Modèle réel chargé depuis : {MODEL_PATH}")
else:
    print("Modèle non trouvé → utilisation d'un modèle mock pour les tests")
    import numpy as np
    
    class MockModel:
        def predict_proba(self, X):
            # Retourne toujours proba fraude = 0.0 pour les tests
            return np.zeros((len(X), 2))  # [[1.0, 0.0], ...] → classe 0 = légitime
        
        @property
        def classes_(self):
            return np.array([0, 1])
    
    model = MockModel()

app = FastAPI(
    title="Détection de Fraude Bancaire - MLOps 2025 🕵️‍♂️",
    description="API XGBoost + DVC + MLflow + Evidently | Dashboard Streamlit",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class Transaction(BaseModel):
    features: list[float] = Field(
        ...,
        description="30 features normalisées (Time, V1 à V28, Amount)",
        example=[0.0] * 30
    )

@app.get("/", tags=["Accueil"])
def home():
    return {"message": "API Fraude en ligne !", "docs": "/docs"}

@app.post("/predict", tags=["Prédiction"])
def predict(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.features], 
                          columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                                   'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                   'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'])
        
        proba = float(model.predict_proba(df)[0, 1])
        
        return {
            "probability_fraud": round(proba, 5),
            "is_fraud": proba > 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")