# test_models.py
import os
import joblib
import numpy as np
from data_loader import load_raw, prepare_features
from utils import compute_metrics

def main():
    # Load the original training and testing data
    train_df, test_df = load_raw()

    # Prepare all feature sets (train/val/test)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_features(
        train_df, test_df,
        downsample_frac=0.30,
        random_state=42,
        noise_strength=0.75
    )

    # Load the trained RandomForest model
    rf = joblib.load("models/random_forest.joblib")

    # Load the trained LinearSVC model
    svc = joblib.load("models/svm_lin.joblib")

    # Evaluate RandomForest on test set
    preds_rf = rf.predict(X_test)
    print("RandomForest test metrics:", compute_metrics(y_test, preds_rf))

    # Evaluate LinearSVC on test set
    preds_svc = svc.predict(X_test)
    print("LinearSVC test metrics:", compute_metrics(y_test, preds_svc))

    # Evaluate neural network if saved model exists
    if os.path.exists("models/nn_model.pt"):
        from nn_inference import load_nn, predict as nn_predict

        # Load the neural network model
        load_nn(input_dim, path="models/nn_model.pt")

        # Make predictions for all test samples (one-by-one)
        nn_preds = np.array([nn_predict(X_test[i].reshape(1, -1)) for i in range(len(X_test))])

        print("NN test metrics:", compute_metrics(y_test, nn_preds))

if __name__ == "__main__":
    main()
