# utils.py
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    """Calculate common evaluation metrics for classification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),            # proportion of correct predictions
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # accuracy on predicted positives
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),        # ability to find all positives
        "f1": float(f1_score(y_true, y_pred, zero_division=0))                  # harmonic mean of precision/recall
    }

def print_metrics_table(result_dict):
    """Print a clean table showing model performance metrics."""
    print("\nEvaluation Results:")
    print("{:<20} {:>8} {:>8} {:>8} {:>8}".format("Model", "Acc", "Prec", "Rec", "F1"))

    for name, m in result_dict.items():                                # loop through each model's results
        print("{:<20} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            name, m["accuracy"], m["precision"], m["recall"], m["f1"]
        ))

def save_joblib(obj, path):
    """Save any Python object (model, scaler, etc.) using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)      # make sure folder exists
    joblib.dump(obj, path)                                 # save object to disk
    print(f"[saved] {path}")                               # confirmation message

def load_joblib(path):
    """Load an object stored with joblib."""
    return joblib.load(path)
