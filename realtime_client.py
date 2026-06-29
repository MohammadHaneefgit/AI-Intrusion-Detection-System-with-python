# realtime_client.py
import socket
import json
import numpy as np
import joblib
import os

HOST = "127.0.0.1"      # server address
PORT = 9999             # server port

def send_sample(features):
    """Send one feature vector to the realtime server and return its response."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))                         # connect to the server

        payload = {"features": list(map(float, features))}  # convert features to list of floats
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))  # send data as JSON

        data = b""                                      # receive buffer
        while True:
            part = s.recv(4096)                         # read server response
            if not part:
                break
            data += part
            if b"\n" in data:                           # stop reading after newline
                break

        msg = data.decode("utf-8").strip()              # decode bytes
        try:
            return json.loads(msg)                      # return parsed JSON
        except:
            return {"error": "invalid response", "raw": msg}  # handle bad output

def send_saved_sample(path="sample_for_realtime.npy"):
    """Load a saved .npy sample and send it to the server."""
    features = np.load(path)                            # load saved features
    return send_sample(features)                        # forward to send_sample()

if __name__ == "__main__":
    print("Sending saved sample...")                    # simple test run
    print(send_saved_sample())                          # print server reply
