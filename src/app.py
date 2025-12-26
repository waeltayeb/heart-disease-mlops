from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__, template_folder="templates")

API_URL = os.environ.get(
    "API_URL",
    "https://heart-disease-api-412775386792.europe-west1.run.app/predict"
)

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
    payload = {f: float(request.form[f]) for f in FEATURES}

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        return "Erreur API", 500

    prediction = response.json()["prediction"]

    result = (
        "❌ Maladie cardiaque détectée"
        if prediction == 1
        else "✅ Pas de maladie cardiaque"
    )

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
