from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def get_models() -> dict[str, object]:
    """Models and tuned parameters reported by the paper."""
    return {
        "LogisticRegression": LogisticRegression(
            C=10,
            l1_ratio=1,
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=5,
            random_state=RANDOM_STATE,
        ),
        "GaussianNB": GaussianNB(var_smoothing=1.232846739442066e-7),
        "RandomForestClassifier": RandomForestClassifier(
            bootstrap=True,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=200,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNeighborsClassifier": KNeighborsClassifier(
            algorithm="auto",
            n_neighbors=9,
            weights="distance",
        ),
    }
