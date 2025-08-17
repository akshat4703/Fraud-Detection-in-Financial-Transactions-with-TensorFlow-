# train_autoencoder.py
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_PATH = os.path.join("models", "autoencoder.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1).values
y = df["Class"].values  # not used for training but needed for evaluation

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build Autoencoder
input_dim = X_train.shape[1]

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="linear")  # reconstruct input
])

autoencoder.compile(optimizer="adam", loss="mse")

# 5. Train Autoencoder (unsupervised, only on non-fraud class = y==0)
X_train_nonfraud = X_train[y_train == 0]

history = autoencoder.fit(
    X_train_nonfraud, X_train_nonfraud,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. Save model
autoencoder.save(MODEL_PATH)

# 7. Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"✅ Autoencoder saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
