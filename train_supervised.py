# train_supervised.py
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier

# Paths
DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_PATH = os.path.join("models", "supervised.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1).values
y = df["Class"].values

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Define Keras model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# 5. Wrap model with SciKeras
clf = KerasClassifier(
    model=create_model,
    epochs=5,
    batch_size=32,
    verbose=1
)

# 6. Train model
clf.fit(X_train, y_train)

# 7. Save trained Keras model
clf.model_.save(MODEL_PATH)

# 8. Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"✅ Supervised model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
