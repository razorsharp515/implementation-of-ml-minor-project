from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH
from .data import prepare_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraud for transactions in a CSV file.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with transaction rows.")
    parser.add_argument("--output", required=True, type=Path, help="Where to save predictions.")
    parser.add_argument("--model", default=MODEL_PATH, type=Path, help="Trained joblib model path.")
    return parser.parse_args()


def align_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    for column in feature_names:
        if column not in frame.columns:
            frame[column] = 0
    return frame[feature_names]


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    df = pd.read_csv(args.input)

    if "isFraud" in df.columns:
        x, _ = prepare_features(df)
    else:
        placeholder = df.copy()
        placeholder["isFraud"] = 0
        x, _ = prepare_features(placeholder)

    x = align_features(x, bundle["feature_names"])
    x_scaled = pd.DataFrame(
        bundle["scaler"].transform(x),
        columns=bundle["feature_names"],
    )

    probabilities = bundle["model"].predict_proba(x_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = df.copy()
    result["fraud_probability"] = probabilities
    result["predicted_isFraud"] = predictions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
