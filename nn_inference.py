# nn_inference.py
import torch
import numpy as np
from nn_model import MediumNN

# Choose device (GPU if available, otherwise CPU)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model reference (loaded once and reused)
_MODEL = None

def load_nn(input_dim, path="models/nn_model.pt"):
    """Load the saved neural network model from disk."""
    global _MODEL
    model = MediumNN(input_dim)                      # create model with correct input size
    model.load_state_dict(torch.load(path, map_location=_DEVICE))  # load weights
    model.to(_DEVICE)                                # move model to correct device
    model.eval()                                     # set to evaluation mode
    _MODEL = model
    return _MODEL

def predict(features, model_path="models/nn_model.pt", input_dim=None):
    """
    Predict using the neural network.
    - features can be a single sample or batch.
    - loads model automatically if not loaded yet.
    """
    global _MODEL

    # Load model if not already loaded
    if _MODEL is None:
        # If input_dim not given, infer from feature shape
        if input_dim is None:
            if hasattr(features, "shape"):
                if len(features.shape) > 1:
                    input_dim = features.shape[1]
                else:
                    input_dim = features.shape[0]
        load_nn(input_dim, path=model_path)

    # Convert features to numpy array
    x = np.array(features, dtype=np.float32)

    # Ensure correct shape
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Convert to torch tensor
    x_t = torch.tensor(x, dtype=torch.float32).to(_DEVICE)

    # Forward pass without gradient tracking
    with torch.no_grad():
        out = _MODEL(x_t).cpu().numpy().reshape(-1)

    # Convert probabilities to 0/1 output
    preds = (out >= 0.5).astype(int)

    # Return single value or array based on input
    if preds.size == 1:
        return int(preds[0])
    return preds

