"""Load the raw data, build the target and make the train/validation/test split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .features import create_engineered_features


def assign_risk(scores):
    return np.select(
        [scores < config.HIGH_RISK_BELOW, scores < config.LOW_RISK_FROM],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )


def load_dataset(path=config.RAW_DATA_PATH, check_labels=True):
    df = pd.read_csv(path)

    y = pd.Series(assign_risk(df[config.TARGET_SOURCE].to_numpy()), name=config.TARGET)

    if check_labels:
        counts = y.value_counts().to_dict()
        if counts != config.EXPECTED_CLASS_COUNTS:
            raise ValueError(
                f"Label counts changed: {counts} != {config.EXPECTED_CLASS_COUNTS}. "
                "Labels or thresholds differ from the original project."
            )

    # Final_Exam_Score defines the target, so it must never be an input
    X = df.drop(columns=config.DROP_COLUMNS + [config.TARGET_SOURCE])
    X = create_engineered_features(X.copy())

    return X, y


def split_dataset(X, y):
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=config.TEST_FRACTION,
        stratify=y,
        random_state=config.SPLIT_SEED,
    )

    val_share_of_rest = config.VAL_FRACTION / (1 - config.TEST_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=val_share_of_rest,
        stratify=y_rest,
        random_state=config.SPLIT_SEED,
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def stratified_sample(X, y, n, seed=config.SPLIT_SEED):
    """Smaller stratified copy of the data, used for quick smoke runs."""
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_s, y_s
