from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import (
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    DEFAULT_DATASET,
    DEMO_DATASET,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    ROC_PATH,
    SHAP_PATH,
    RANDOM_STATE,
)
from .data import build_train_test_data, create_demo_dataset, load_dataset
from .models import get_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to PaySim/Kaggle CSV. Defaults to data/raw/online_payments.csv if present, otherwise demo data.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plot generation for faster runs.",
    )
    return parser.parse_args()


def choose_dataset(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    return create_demo_dataset(DEMO_DATASET)


def evaluate_models(x_train, y_train, folds: int) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for name, model in get_models().items():
        scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "folds": folds,
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
            }
        )
    return rows


def plot_roc(best_model, x_test, y_test) -> None:
    probabilities = best_model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve for best fraud model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=160)
    plt.close()


def plot_confusion_matrix(best_model, x_test, y_test) -> None:
    predictions = best_model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not fraud", "Fraud"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()


def try_plot_shap(best_model, x_train, skip: bool) -> None:
    if skip:
        return
    try:
        import shap

        sample = x_train.sample(min(300, len(x_train)), random_state=RANDOM_STATE)
        explainer = shap.Explainer(best_model, sample)
        values = explainer(sample)
        shap.summary_plot(values, sample, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(SHAP_PATH, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # SHAP is helpful but not required for training.
        print(f"SHAP plot skipped: {exc}")


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = choose_dataset(args.data)
    df = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test, scaler = build_train_test_data(df)

    rows = []
    for folds in CV_FOLDS:
        rows.extend(evaluate_models(x_train, y_train, folds))

    metrics = pd.DataFrame(rows)
    best_row = metrics.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    best_model_name = str(best_row["model"])
    best_model = clone(get_models()[best_model_name])
    best_model.fit(x_train, y_train)

    predictions = best_model.predict(x_test)
    holdout_row = {
        "model": best_model_name,
        "folds": "holdout",
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1]),
    }
    metrics = pd.concat([metrics, pd.DataFrame([holdout_row])], ignore_index=True)
    metrics.to_csv(METRICS_PATH, index=False)

    plot_roc(best_model, x_test, y_test)
    plot_confusion_matrix(best_model, x_test, y_test)
    try_plot_shap(best_model, x_train, args.skip_shap)

    joblib.dump(
        {
            "model_name": best_model_name,
            "model": best_model,
            "scaler": scaler,
            "feature_names": list(x_train.columns),
            "dataset_path": str(dataset_path),
        },
        MODEL_PATH,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
