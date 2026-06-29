# final.py
import os
import time
import threading
import numpy as np
from data_loader import load_raw, prepare_features
from eda import run_eda
from ml_models import run_ml_pipeline
from nn_model import train_nn
from utils import save_joblib, print_metrics_table
import realtime_server, realtime_client
import socket

def is_port_in_use(port):
    """Check whether the given TCP port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def main():
    # Make sure required folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("outputs", "eda"), exist_ok=True)

    # 1) Load raw CSV files
    train_df, test_df = load_raw()

    # 2) Run EDA (class distribution + summary stats + heatmap)
    run_eda(train_df, test_df, out_dir=os.path.join("outputs", "eda"))

    # 3) Prepare datasets
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_features(
        train_df, test_df,
        downsample_frac=0.30,
        random_state=42,
        noise_strength=0.75
    )

    # Save the scaler for realtime prediction
    save_joblib(scaler, os.path.join("models", "scaler.joblib"))

    # 4) Train classical ML models (RandomForest + LinearSVC)
    ml_results = run_ml_pipeline(X_train, y_train, X_val, y_val, save_dir="models")
    print_metrics_table(ml_results)       # print metrics in clean table

    # 5) Train the neural network and log validation results
    nn_model, nn_metrics = train_nn(
        X_train, y_train, X_val, y_val,
        epochs=8,                 # short training time
        batch_size=256,
        lr=1e-3,
        save_path="models/nn_model.pt"
    )
    print("NN metrics (val):", nn_metrics)

    # 6) Save one pre-scaled training sample for realtime test
    sample = X_train[0].reshape(-1)        # take first sample
    np.save("sample_for_realtime.npy", sample)

    # 7) Reload ML model for server before running client demo
    print("Reloading model for server...")
    realtime_server.load_model()

    # Start realtime server only if port is free
    PORT = 9999
    if is_port_in_use(PORT):
        print(f"Port {PORT} is already in use — server already running. Skipping server startup.")
    else:
        server_thread = threading.Thread(
            target=realtime_server.start_server,
            kwargs={"host": "127.0.0.1", "port": PORT},
            daemon=True
        )
        server_thread.start()
        time.sleep(1.0)   # allow server to start properly

    # 8) Send sample to realtime server and print output
    print("Sending realtime sample...")
    resp = realtime_client.send_saved_sample("sample_for_realtime.npy")
    print("Realtime demo response:", resp)

    # 9) List generated models
    print("Saved ML & NN models in:", os.path.abspath("models"))
    print("Files:", os.listdir("models"))

if __name__ == "__main__":
    main()
