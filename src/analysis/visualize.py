import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os

# Chargement des données et du modèle
X_test = pd.read_csv("data/processed_test.csv")
y_test = pd.read_csv("data/y_test.csv")["Class"]
model_path = "models/xgb_fraud.pkl"
model = joblib.load(model_path)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

os.makedirs("reports", exist_ok=True)

# 1. Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC - Détection de Fraude')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("reports/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fraude', 'Fraude'])
disp.plot(cmap='Blues')
plt.title('Matrice de confusion')
plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Distribution des probabilités
plt.figure(figsize=(10, 6))
pd.Series(y_pred_proba[y_test == 0]).hist(alpha=0.7, bins=50, label='Non-Fraude', color='skyblue')
pd.Series(y_pred_proba[y_test == 1]).hist(alpha=0.7, bins=50, label='Fraude', color='salmon')
plt.title('Distribution des probabilités prédites par classe')
plt.xlabel('Probabilité de fraude')
plt.ylabel('Nombre de transactions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("reports/proba_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("3 graphiques générés avec succès dans reports/ :")
print("   → roc_curve.png")
print("   → confusion_matrix.png")
print("   → proba_distribution.png")