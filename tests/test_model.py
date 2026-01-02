import os
import pytest
import joblib

MODEL_PATH = "models/xgb_fraud.pkl"

# Skip les tests qui nécessitent le modèle si le fichier n'existe pas (cas CI)
requires_model = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Modèle réel non disponible (CI)")

@requires_model
def test_model_exists():
    assert os.path.exists(MODEL_PATH), f"Modèle manquant : {MODEL_PATH}"

@requires_model
def test_prediction_output():
    model = joblib.load(MODEL_PATH)
    # Test simple avec un input de 30 zéros
    sample = [[0.0] * 30]
    proba = model.predict_proba(sample)[0, 1]
    assert 0 <= proba <= 1, "Probabilité doit être entre 0 et 1"