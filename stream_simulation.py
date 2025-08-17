import pandas as pd
import time
import requests
import random
import os

# API URL (FastAPI running locally)
API_URL = "http://127.0.0.1:8000/predict"

# Path to dataset
DATA_PATH = os.path.join("data", "creditcard.csv")

def stream_transactions():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop labels if present
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
        y = df["Class"]
    else:
        X = df
        y = None

    print("🚀 Starting transaction stream...")
    for i in range(20):  # stream only 20 transactions for demo
        # Pick a random transaction
        idx = random.randint(0, len(X) - 1)
        transaction = X.iloc[idx].tolist()

        # Send to API
        response = requests.post(API_URL, json={"features": transaction})

        if response.status_code == 200:
            result = response.json()
            print(f"Transaction {i+1}:")
            print(f"  True Label: {y.iloc[idx] if y is not None else 'N/A'}")
            print(f"  Supervised → Prob: {result['supervised']['probability']:.4f}, Pred: {result['supervised']['prediction']}")
            print(f"  Autoencoder → Error: {result['autoencoder']['reconstruction_error']:.6f}, Pred: {result['autoencoder']['prediction']}")
            print("-" * 60)
        else:
            print(f"⚠️ API Error: {response.text}")

        # Sleep to simulate real-time stream
        time.sleep(1)


if __name__ == "__main__":
    stream_transactions()
