# ml_models.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from utils import compute_metrics, save_joblib


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=2,        # very small number of trees
        max_depth=3,          # shallow depth to reduce memorisation
        min_samples_leaf=40,  # require more samples per leaf
        min_samples_split=40, # require more samples per split
        max_features=2,       # limit number of features used
        criterion="entropy",  # alternative split measure
        bootstrap=False,      # no bootstrapping
        class_weight="balanced_subsample",  # balance classes
        random_state=42       # keep training reproducible
    )
    rf.fit(X_train, y_train)  # train the model
    return rf


def train_linearsvc(X_train, y_train):
    rng = np.random.RandomState(42)  # random generator for noise

    # Flip a small percentage of labels to add noise
    y_train_corrupted = y_train.copy()  # copy labels
    flip_mask = rng.rand(len(y_train)) < 0.02  # 2% flip chance
    y_train_corrupted[flip_mask] = 1 - y_train_corrupted[flip_mask]  # flip 0↔1

    n_features = X_train.shape[1]
    keep = max(3, int(n_features * 0.40))   # number of features to use
    feature_ids = rng.choice(n_features, size=keep, replace=False)  # random subset
    X_train_small = X_train[:, feature_ids]  # feature matrix

    # Train LinearSVC with small C for regularisation
    svc = LinearSVC(
        C=0.02,                 # strong regularisation
        loss="squared_hinge",   # default loss type
        max_iter=3000,          # enough iterations to converge
        tol=1e-3,               # tolerance level
        class_weight="balanced",# handle imbalance
        dual=False,             # faster when features > samples
        random_state=42         # reproducible training
    )
    svc.fit(X_train_small, y_train_corrupted)  # train on corrupted subset

    svc.feature_ids_ = feature_ids  # keep track of used features
    return svc

# Evaluate model performance on test or validation sets
def evaluate_model(model, X_test, y_test):
    if hasattr(model, "feature_ids_"):
        X_test_eval = X_test[:, model.feature_ids_]  # select same features as training
    else:
        X_test_eval = X_test  # full feature set if RF

    preds = model.predict(X_test_eval)  # get predictions
    return compute_metrics(y_test, preds), preds  # return metrics & predictions

# Full ML training pipeline for RF and SVC
def run_ml_pipeline(X_train, y_train, X_val, y_val, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)  # ensure model directory exists
    results = {}  # store evaluation

    # Train RandomForest
    rf = train_random_forest(X_train, y_train)
    rf_metrics, _ = evaluate_model(rf, X_val, y_val)  # evaluate
    results["RandomForest"] = rf_metrics
    save_joblib(rf, os.path.join(save_dir, "random_forest.joblib"))  # save model

    # Train LinearSVC
    svc = train_linearsvc(X_train, y_train)
    svc_metrics, _ = evaluate_model(svc, X_val, y_val)  # evaluate
    results["LinearSVC"] = svc_metrics
    save_joblib(svc, os.path.join(save_dir, "svm_lin.joblib"))  # save model

    return results  # return final results
