# generate_shap.py
import os
import pickle
import shap
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_PATH = os.path.join("models", "supervised.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")
SHAP_SUMMARY_PATH = os.path.join("models", "shap_summary.png")
SHAP_BAR_PATH = os.path.join("models", "shap_bar.png")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1).values
y = df["Class"].values
feature_names = df.drop("Class", axis=1).columns

# 2. Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# 3. Load supervised model
model = tf.keras.models.load_model(MODEL_PATH)

# 4. Create SHAP explainer
explainer = shap.Explainer(model, X_scaled[:500])  # use small background for speed
shap_values = explainer(X_scaled[:500])

# 5. Summary plot
plt.figure()
shap.summary_plot(shap_values, features=X_scaled[:500], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(SHAP_SUMMARY_PATH)
plt.close()

# 6. Bar plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(SHAP_BAR_PATH)
plt.close()

print(f"✅ SHAP summary saved to {SHAP_SUMMARY_PATH}")
print(f"✅ SHAP bar plot saved to {SHAP_BAR_PATH}")
