# 🚨 Fraud Detection in Financial Transactions (TensorFlow + FastAPI + SHAP)

This project demonstrates fraud detection in financial transactions using both **Supervised Learning** and **Autoencoders**.  
It also provides **explainability with SHAP**, a **REST API with FastAPI**, and a **transaction streaming simulator**.

---

## 📂 Project Structure
fraud-detection/
│── data/
│ └── creditcard.csv # Dataset (Kaggle Credit Card Fraud Detection)
│── models/
│ └── supervised.keras # Supervised (MLP) trained model
│ └── autoencoder.keras # Autoencoder trained model
│ └── scaler.pkl # StandardScaler for preprocessing
│ └── shap_summary.png # SHAP summary plot
│ └── shap_bar.png # SHAP feature importance bar chart
│── api/
│ └── app.py # FastAPI app for serving fraud detection
│── train_supervised.py # Train supervised model
│── train_autoencoder.py # Train autoencoder
│── generate_shap.py # Generate SHAP plots
│── stream_simulation.py # Simulate live transaction streaming
│── requirements.txt # Project dependencies

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akshat4703/fraud-detection.git
   cd fraud-detection

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

📊 Dataset

The dataset is from Kaggle Credit Card Fraud Detection.
Place creditcard.csv inside the data/ folder.

🧠 Training
1. Train Supervised Model
   ```bash
   python train_supervised.py

2. Train Autoencoder Model
   ```bash
   python train_autoencoder.py

3. Generate SHAP Plots
   ```bash
   python generate_shap.py

🚀 Running the API

Start the FastAPI server:
'''bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

Server runs at:
http://127.0.0.1:8000

Test docs at:
http://127.0.0.1:8000/docs

📡 Simulate Live Transactions

Keep the API server running, then in a new terminal:
'''bash
python stream_simulation.py
This script streams random transactions to the API and prints fraud predictions.

📈 Explainability with SHAP

shap_summary.png → SHAP summary plot

shap_bar.png → SHAP feature importance

These help understand why the model flagged a transaction as fraud.

🛠️ Requirements

Main dependencies:

Python 3.10+

TensorFlow 2.16+

scikit-learn 1.5+

shap

matplotlib

FastAPI

Uvicorn

SciKeras

Install all with:
'''bash
pip install -r requirements.txt

✨ Features

✅ Supervised fraud detection (MLP classifier)

✅ Unsupervised fraud detection (Autoencoder)

✅ Explainability with SHAP

✅ REST API with FastAPI

✅ Real-time transaction streaming simulation



