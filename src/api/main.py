from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import joblib

from src.api.schemas import HeartDiseaseInput, PredictionResponse

app = FastAPI(title="Heart Disease Prediction API")

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
MONITORING_DATA = BASE_DIR / "monitoring" / "data" / "current_data.csv"

# =========================
# Load model
# =========================
model = joblib.load(MODELS_DIR / "model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

# =========================
# Feature schema (IMPORTANT)
# =========================
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

CSV_COLUMNS = FEATURES + ["target", "prediction"]

# =========================
# Endpoint
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartDiseaseInput):
    # Input → DataFrame
    df = pd.DataFrame([data.dict()])

    # Prediction
    df_scaled = scaler.transform(df[FEATURES])
    prediction = int(model.predict(df_scaled)[0])

    # Monitoring fields
    df["target"] = -1          # inconnu en production
    df["prediction"] = prediction

    # Enforce column order (CRUCIAL)
    df = df[CSV_COLUMNS]

    # Create folder if needed
    MONITORING_DATA.parent.mkdir(parents=True, exist_ok=True)

    # Append to CSV
    df.to_csv(
        MONITORING_DATA,
        mode="a",
        header=not MONITORING_DATA.exists(),
        index=False
    )

    return PredictionResponse(prediction=prediction)
