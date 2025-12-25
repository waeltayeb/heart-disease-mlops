import pandas as pd
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

REFERENCE_PATH = DATA_DIR / "reference_data.csv"
CURRENT_PATH = DATA_DIR / "current_data.csv"

REPORT_HTML = REPORTS_DIR / "monitoring_report.html"
REPORT_JSON = REPORTS_DIR / "monitoring_report.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load data
# =========================
reference = pd.read_csv(REFERENCE_PATH)
current = pd.read_csv(CURRENT_PATH)

# =========================
# Column mapping (IGNORE prediction)
# =========================
column_mapping = ColumnMapping(
    target=None,
    prediction=None
)

# =========================
# Evidently Drift ONLY
# =========================
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=reference,
    current_data=current,
    column_mapping=column_mapping
)

# =========================
# Save report
# =========================
report.save_html(str(REPORT_HTML))
report.save_json(str(REPORT_JSON))

print("✅ Monitoring Data Drift (PRODUCTION) terminé")
print(f"📄 HTML : {REPORT_HTML}")
print(f"📄 JSON : {REPORT_JSON}")
