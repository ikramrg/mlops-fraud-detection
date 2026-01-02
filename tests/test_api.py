import os
import sys
# Ajoute la racine du projet au path pour que 'src' soit importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    sample_transaction = {
        "features": [0.0] * 30  # 30 zéros = transaction normale
    }
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    data = response.json()
    assert "probability_fraud" in data
    assert "is_fraud" in data
    assert isinstance(data["probability_fraud"], float)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()