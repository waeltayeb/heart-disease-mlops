from flask import Flask, render_template, request
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("models/model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = []
    for feature in FEATURES:
        value = float(request.form[feature])
        data.append(value)

    input_data = np.array(data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    result = (
        "Maladie cardiaque détectée"
        if prediction == 0
        else "✅ Pas de maladie cardiaque"
    )

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
