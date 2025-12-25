import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# ==============================
# PATHS
# ==============================
DATA_PATH = "data/heart_disease_data.csv"
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_DIR = "monitoring"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD REFERENCE DATA
# ==============================
df = pd.read_csv(DATA_PATH)
X_ref = df.drop("target", axis=1)

# ==============================
# SIMULATE PRODUCTION DATA
# ==============================
X_prod = X_ref.sample(frac=0.3, random_state=42)
X_prod = X_prod + 0.05 * X_ref.std()

# ==============================
# PREPARE EVIDENTLY DATA
# ==============================
reference_data = X_ref.copy()
production_data = X_prod.copy()

# ==============================
# CREATE REPORT
# ==============================
report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference_data,
    current_data=production_data
)

# ==============================
# SAVE REPORTS
# ==============================
html_path = os.path.join(OUTPUT_DIR, "data_drift_report.html")
json_path = os.path.join(OUTPUT_DIR, "data_drift_report.json")

report.save_html(html_path)
report.save_json(json_path)

print("✅ Monitoring terminé avec succès")
print(f"📄 HTML report : {html_path}")
print(f"📄 JSON report : {json_path}")
