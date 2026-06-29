# data_loader.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_raw(train_filename="data/KDDTrain.csv", test_filename="data/KDDTest.csv"):
    """Load training and testing CSV files."""
    train_path = os.path.join(os.getcwd(), train_filename)  # build full path for train file
    test_path = os.path.join(os.getcwd(), test_filename)    # build full path for test file

    print("Loading:", train_path)
    print("Loading:", test_path)

    train_df = pd.read_csv(train_path)  # read training CSV
    test_df = pd.read_csv(test_path)    # read testing CSV

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, test_df  # return loaded dataframes


def _safe_label_encode_train(train_series):
    """Fit a LabelEncoder only on the training column."""
    le = LabelEncoder()  # create label encoder
    le.fit(train_series.astype(str).values)  # learn categories from training data only
    return le  # return fitted encoder


def _safe_transform_with_map(le, series):
    """Transform categories using a safe map (unseen values become -1)."""
    mapping = {v: i for i, v in enumerate(le.classes_)}  # create mapping for known labels
    return series.astype(str).map(lambda x: mapping.get(x, -1)).astype(int).values  # map with fallback -1


def prepare_features(train_df, test_df, downsample_frac=0.30, random_state=42, noise_strength=0.75):
    """
    Preprocess data and return all feature/label splits plus scaler and input dimension.
    """

    train_df = train_df.copy()  # work on a copy to avoid modifying original
    test_df = test_df.copy()    # same for test data

    # Downsample training set to reduce duplicates
    if 0 < downsample_frac < 1.0:
        train_df = train_df.sample(frac=downsample_frac, random_state=random_state).reset_index(drop=True)  # sample rows

    # Encode the label column (normal / attack)
    label_le = LabelEncoder()  # create encoder for labels
    train_df["label"] = label_le.fit_transform(train_df["label"].astype(str))  # fit+transform on train
    test_df["label"] = label_le.transform(test_df["label"].astype(str))        # transform test using same encoder

    # Encode any categorical columns (protocol, service, flag, etc.)
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()  # find categorical columns
    for col in cat_cols:
        le_col = _safe_label_encode_train(train_df[col])  # fit encoder on training column
        train_df[col] = _safe_transform_with_map(le_col, train_df[col])  # transform train values
        test_df[col] = _safe_transform_with_map(le_col, test_df[col])    # transform test values

    # Separate features and labels
    X_full = train_df.drop(columns=["label"])  # all training features
    y_full = train_df["label"].values          # training labels

    X_test = test_df.drop(columns=["label"])   # test features
    y_test = test_df["label"].values           # test labels

    # Create validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=random_state, stratify=y_full  # stratify to keep class balance
    )

    # Standard scaling (fit only on training)
    scaler = StandardScaler()    # create scaler
    X_train = scaler.fit_transform(X_train)  # fit+transform training data
    X_val = scaler.transform(X_val)          # scale validation data
    X_test = scaler.transform(X_test)        # scale test data

    rng = np.random.RandomState(random_state)  # random generator
    if noise_strength is not None and noise_strength > 0.0:
        X_train = X_train + rng.normal(0, noise_strength, size=X_train.shape)  # add noise to train
        X_val   = X_val   + rng.normal(0, noise_strength, size=X_val.shape)    # add noise to validation
        X_test  = X_test  + rng.normal(0, noise_strength, size=X_test.shape)   # add noise to test

    input_dim = X_train.shape[1]  # number of features for ML/NN models

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim  # return all processed items
