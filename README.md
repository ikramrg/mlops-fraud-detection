# MLOps Fraud Detection ??????

Détection de fraude bancaire avec XGBoost.

## Stack
- XGBoost + SMOTE
- DVC (versioning données & pipeline)
- MLflow (tracking)
- Evidently (drift monitoring)
- FastAPI (API)
- Docker
- GitHub Actions (CI/CD)

## Lancer le projet
`ash
pip install -r requirements.txt
dvc repro
uvicorn src.api.main:app --reload
``n
Docs API : http://localhost:8000/docs
