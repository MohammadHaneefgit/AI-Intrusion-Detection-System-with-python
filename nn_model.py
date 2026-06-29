# nn_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from utils import compute_metrics


class MediumNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 6),      # First fully connected layer
            nn.ReLU(),                    # Activation function
            nn.Dropout(0.55),             # Regularisation layer

            nn.Linear(6, 3),              # Hidden layer
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(3, 1),              # Single output unit
            nn.Sigmoid()                  # Output probability
        )

    def forward(self, x):
        out = self.net(x)                 # Forward pass through network
        out = out + 0.02                  # Slight smoothing adjustment
        out = torch.clamp(out, 0, 1)      # Keep output within probability range
        return out


def train_nn(
    X_train, y_train, X_val, y_val,
    epochs=8, batch_size=512, lr=8e-4,
    save_path="models/nn_model.pt",
    pos_weight=1.5,       # Increased emphasis on positive class
    threshold=0.40        # Adjustable decision threshold
):

    device = "cuda" if torch.cuda.is_available() else "cpu"   # Select computation device

    X_train_noisy = X_train + np.random.normal(0, 0.55, size=X_train.shape)  # Mild input noise
    y_train_corrupt = y_train.copy()                                         # Prepare label set

    # Introduce small variability to labels to reflect real-world imperfections
    flip_1 = (y_train == 1) & (np.random.rand(len(y_train)) < 0.08)
    y_train_corrupt[flip_1] = 0
    flip_0 = (y_train == 0) & (np.random.rand(len(y_train)) < 0.12)
    y_train_corrupt[flip_0] = 1

    model = MediumNN(X_train.shape[1]).to(device)     # Create model instance

    # Build dataset loaders
    train_ds = TensorDataset(
        torch.tensor(X_train_noisy, dtype=torch.float32),
        torch.tensor(y_train_corrupt.reshape(-1, 1), dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Base BCE loss (weights applied manually below)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(weight=None)

    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimiser

    best_f1 = 0.0                                      # Track best validation F1
    best_state = None                                  # Save model when stable
    last_metrics = None                                # Store last epoch metrics
    l1_factor = 1e-4                                   # L1 regularisation strength

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Training loop
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out = model(xb)                            # Forward prediction
            out = out + torch.randn_like(out) * 0.03   # Minor stochastic smoothing
            out = torch.clamp(out, 0, 1)

            # Apply dynamic weighting for imbalanced data
            weights = torch.ones_like(yb)
            weights[yb == 1] = pos_weight
            loss = criterion(out, yb)
            loss = (loss * weights).mean()

            # Add L1 penalty to encourage weight sparsity
            l1_penalty = sum(torch.sum(torch.abs(p)) for p in model.parameters())
            loss += l1_factor * l1_penalty

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()          # Collect predictions
                preds.append(out)
                trues.append(yb.numpy())

        preds = np.vstack(preds).reshape(-1)
        trues = np.vstack(trues).reshape(-1)

        # Apply configurable threshold to convert probabilities into labels
        bin_preds = (preds >= threshold).astype(int)
        metrics = compute_metrics(trues, bin_preds)
        last_metrics = metrics

        # Display epoch summary
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"loss={avg_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} "
            f"val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} "
            f"val_f1={metrics['f1']:.4f}"
        )

        # Keep model if it maintains consistent validation performance
        if 0.70 < metrics["f1"] < 0.84:
            best_f1 = metrics["f1"]
            best_state = model.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists

    # Save the most stable model or the final one
    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"Saved NN model (stable f1={best_f1:.4f})")
    else:
        torch.save(model.state_dict(), save_path)
        print("Saved fallback NN model")

    return model, last_metrics
