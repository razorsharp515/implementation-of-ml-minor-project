from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from .config import (
    DEMO_DATASET,
    DROP_COLUMNS,
    RANDOM_STATE,
    SAMPLE_DATA_DIR,
    TARGET_COLUMN,
    TRANSACTION_TYPES,
)


def create_demo_dataset(path: Path = DEMO_DATASET, rows: int = 6000) -> Path:
    """Create a small PaySim-like dataset so the project runs immediately."""
    rng = np.random.default_rng(RANDOM_STATE)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    transaction_type = rng.choice(
        TRANSACTION_TYPES,
        rows,
        p=[0.22, 0.34, 0.04, 0.30, 0.10],
    )
    amount = rng.lognormal(mean=9.0, sigma=1.1, size=rows).round(2)
    oldbalance_org = rng.lognormal(mean=10.0, sigma=1.0, size=rows).round(2)

    risky_type = np.isin(transaction_type, ["TRANSFER", "CASH_OUT"])
    empties_origin = amount > oldbalance_org * rng.uniform(0.75, 1.25, rows)
    fraud_probability = 0.006 + risky_type * 0.08 + empties_origin * 0.08
    is_fraud = rng.binomial(1, np.clip(fraud_probability, 0, 0.85))

    newbalance_orig = np.where(
        is_fraud == 1,
        0,
        np.maximum(oldbalance_org - amount, 0),
    ).round(2)
    oldbalance_dest = rng.lognormal(mean=9.8, sigma=1.2, size=rows).round(2)
    newbalance_dest = np.where(
        np.isin(transaction_type, ["TRANSFER", "CASH_OUT"]),
        oldbalance_dest + amount,
        oldbalance_dest,
    ).round(2)
    is_flagged = ((amount > 200000) & (transaction_type == "TRANSFER")).astype(int)

    df = pd.DataFrame(
        {
            "step": rng.integers(1, 745, rows),
            "type": transaction_type,
            "amount": amount,
            "nameOrig": [f"C{i:07d}" for i in range(rows)],
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "nameDest": [f"{'M' if i % 5 == 0 else 'C'}{i + 9000000}" for i in range(rows)],
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = DEMO_DATASET
    path = Path(path)
    if not path.exists() and path == DEMO_DATASET:
        create_demo_dataset(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")

    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN])
    x = x.drop(columns=[c for c in DROP_COLUMNS if c in x.columns])

    if "type" in x.columns:
        x = pd.get_dummies(x, columns=["type"], prefix="type", dtype=int)

    for transaction_type in TRANSACTION_TYPES:
        column = f"type_{transaction_type}"
        if column not in x.columns:
            x[column] = 0

    x = x.select_dtypes(include=["number", "bool"]).astype(float)
    return x, y


def build_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    x, y = prepare_features(df)

    sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    x_balanced, y_balanced = sampler.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x_balanced,
        y_balanced,
        test_size=test_size,
        stratify=y_balanced,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
