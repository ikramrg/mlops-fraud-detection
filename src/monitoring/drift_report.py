import pandas as pd
import joblib
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
import os

ref = pd.read_csv("data/processed_train.csv")
curr = pd.read_csv("data/processed_test.csv")

report = Report(metrics=[DataDriftTable(), DatasetDriftMetric()])
report.run(reference_data=ref, current_data=curr)
os.makedirs("reports", exist_ok=True)
report.save_html("reports/drift_report.html")
print("Rapport de drift généré → reports/drift_report.html")