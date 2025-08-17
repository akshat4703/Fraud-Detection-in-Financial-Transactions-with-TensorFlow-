from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import os

# Paths
SUPERVISED_MODEL_PATH = os.path.join("models", "supervised.keras")
AUTOENCODER_MODEL_PATH = os.path.join("models", "autoencoder.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Initialize app
app = FastAPI(title="Fraud Detection API", description="Fraud detection using Supervised + Autoencoder", version="1.0")

# Load models & scaler at startup
supervised_model = tf.keras.models.load_model(SUPERVISED_MODEL_PATH)
autoencoder_model = tf.keras.models.load_model(AUTOENCODER_MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Compute autoencoder threshold (can be tuned based on validation set)
# For simplicity, set a fixed threshold (adjust if needed)
THRESHOLD = 0.01  

# Request schema
class Transaction(BaseModel):
    features: list  # transaction features (same order as training data)


@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to numpy array
    data = np.array(transaction.features).reshape(1, -1)

    # Scale features
    data_scaled = scaler.transform(data)

    # Supervised prediction
    prob = supervised_model.predict(data_scaled, verbose=0)[0][0]
    supervised_pred = int(prob > 0.5)

    # Autoencoder prediction
    recon = autoencoder_model.predict(data_scaled, verbose=0)
    error = mean_squared_error(data_scaled, recon)
    autoencoder_pred = int(error > THRESHOLD)

    return {
        "supervised": {"probability": float(prob), "prediction": supervised_pred},
        "autoencoder": {"reconstruction_error": float(error), "prediction": autoencoder_pred}
    }
