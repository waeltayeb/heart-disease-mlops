import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# CONFIGURATION MLflow
# ==============================
MLRUNS_PATH = "./mlruns"

mlflow.set_tracking_uri(f"file:{os.path.abspath(MLRUNS_PATH)}")
mlflow.set_experiment("Heart Disease Prediction")

# ==============================
# PATHS
# ==============================
DATA_PATH = "data/heart_disease_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
print("📥 Chargement du dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# MODEL
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Training terminé — Accuracy: {accuracy:.4f}")

# ==============================
# MLflow LOGGING
# ==============================
with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log scaler as artifact
    mlflow.log_artifact(SCALER_PATH if os.path.exists(SCALER_PATH) else MODEL_DIR)

# ==============================
# SAVE LOCAL MODEL
# ==============================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("💾 Modèle et scaler sauvegardés dans /models")
print("📊 MLflow tracking activé — dossier mlruns créé")
