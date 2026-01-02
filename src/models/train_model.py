import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.xgboost
import joblib
import yaml
import os
import matplotlib.pyplot as plt

# Chargement des paramètres
params_path = os.path.join(os.path.dirname(__file__), "..", "..", "params.yaml")
with open(params_path) as f:
    params = yaml.safe_load(f)

# Chargement et split
df = pd.read_csv(params["data"]["raw_data"])
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["features"]["test_size"],
    random_state=params["features"]["random_state"],
    stratify=y
)

# Sauvegarde des données traitées
X_train.to_csv(params["data"]["train_data"], index=False)
X_test.to_csv(params["data"]["test_data"], index=False)
pd.DataFrame(y_train, columns=["Class"]).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["Class"]).to_csv("data/y_test.csv", index=False)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Entraînement avec MLflow
mlflow.set_experiment("fraud-detection")
with mlflow.start_run(run_name="xgboost_smote"):
    model = xgb.XGBClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        learning_rate=params["model"]["learning_rate"],
        scale_pos_weight=params["model"]["scale_pos_weight"],
        eval_metric="aucpr",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Métriques
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    mlflow.log_params(params["model"])
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.log_metric("average_precision", ap)

    # === GRAPHIQUES DANS MLFLOW UI ===
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend()
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fraude', 'Fraude'])
    disp.plot(cmap='Blues')
    plt.title('Matrice de confusion')
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(10,6))
    pd.Series(y_pred_proba[y_test == 0]).hist(alpha=0.7, bins=50, label='Non-Fraude', color='skyblue')
    pd.Series(y_pred_proba[y_test == 1]).hist(alpha=0.7, bins=50, label='Fraude', color='salmon')
    plt.title('Distribution des probabilités par classe')
    plt.xlabel('Probabilité de fraude')
    plt.ylabel('Fréquence')
    plt.legend()
    mlflow.log_figure(plt.gcf(), "proba_distribution.png")
    plt.close()

    # === SAUVEGARDE DU MODÈLE AU BON ENDROIT (pour DVC) ===
    os.makedirs("../../models", exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "xgb_fraud.pkl")
    joblib.dump(model, model_path)
    mlflow.xgboost.log_model(model, "model")
    mlflow.log_artifact(params_path, artifact_path="config")

print(f"ROC AUC = {auc_score:.4f} | Average Precision = {ap:.4f}")
print("Modèle + graphiques sauvegardés dans MLflow UI !")