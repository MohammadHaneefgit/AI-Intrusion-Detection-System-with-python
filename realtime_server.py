# realtime_server.py
import socket
import threading
import json
import numpy as np
import joblib
import os
from nn_inference import predict as nn_predict
from utils import load_joblib

HOST = "127.0.0.1"                   # server address
PORT = 9999                          # server port
USE_NN = False                       # toggle between ML model and NN

MODEL_PATH = os.path.join("models", "random_forest.joblib")  # path to ML model
_MODEL = None                        # global model holder
_SCALER = None                       # global scaler holder

def load_model():
    """Load the ML model and scaler for the server."""
    global _MODEL, _SCALER

    try:
        if os.path.exists(MODEL_PATH):                     # check if model exists
            _MODEL = load_joblib(MODEL_PATH)               # load trained model

            scaler_path = os.path.join("models", "scaler.joblib")
            if os.path.exists(scaler_path):                # load scaler if available
                _SCALER = load_joblib(scaler_path)

            print("Server model loaded successfully!")
            return True
        else:
            print("Server: model file not present yet:", MODEL_PATH)
            return False

    except Exception as e:
        print("Reloading model for server failed:", e)
        _MODEL = None
        return False

def handle_client(conn, addr):
    """Handle each incoming client connection."""
    print("Connected by", addr)
    buffer = b""                                          # receive buffer

    try:
        while True:
            data = conn.recv(4096)                         # receive data
            if not data:
                break

            buffer += data
            if b"\n" in buffer:                           # process line-based messages
                lines = buffer.split(b"\n")

                for line in lines[:-1]:
                    try:
                        payload = json.loads(line.decode("utf-8"))  # parse JSON
                        features = np.array(payload["features"], dtype=float).reshape(1, -1)

                        if USE_NN:
                            pred = nn_predict(features, input_dim=features.shape[1])   # NN prediction
                        else:
                            if _MODEL is None:                                  # load model on first request
                                ok = load_model()
                                if not ok:
                                    resp = {"error": "no model loaded"}
                                    conn.sendall((json.dumps(resp) + "\n").encode())
                                    continue

                            pred = int(_MODEL.predict(features)[0])              # ML model prediction

                        resp = {"prediction": int(pred)}                         # prepare response

                    except Exception as e:
                        resp = {"error": str(e)}                                 # send error message

                    conn.sendall((json.dumps(resp) + "\n").encode())             # send reply

                buffer = lines[-1]                                              # carry over remainder

    finally:
        conn.close()
        print("Disconnected", addr)

def start_server(host="127.0.0.1", port=9999):
    """Start the TCP server and listen for incoming clients."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))                                 # bind port
        s.listen()                                           # start listening
        print(f"Server listening on {host}:{port}")

        while True:
            conn, addr = s.accept()                          # accept incoming client
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()                                        # handle clients on separate threads

if __name__ == "__main__":
    load_model()                                             # try loading model before starting
    start_server(HOST, PORT)                                 # start server
