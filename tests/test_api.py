from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    sample_transaction = {
        "features": [0] * 30  # ou un exemple réel
    }
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    assert "probability_fraud" in response.json()
    assert "is_fraud" in response.json()