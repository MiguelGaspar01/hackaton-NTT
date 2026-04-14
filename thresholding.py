from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, recall_score


@dataclass
class ThresholdedClassifier:
    """Classifier wrapper that applies a fixed decision threshold on class-1 probability."""

    base_model: object
    threshold: float = 0.5

    def fit(self, X, y, **fit_params):
        self.base_model.fit(X, y, **fit_params)
        return self

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        y_prob = self.predict_proba(X)[:, 1]
        return (y_prob >= self.threshold).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def __getattr__(self, name):
        # Delegate unknown attributes to the wrapped estimator.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        if name == "base_model":
            raise AttributeError(name)

        base_model = self.__dict__.get("base_model")
        if base_model is None:
            raise AttributeError(name)

        return getattr(base_model, name)


def choose_best_threshold_for_recall(
    y_true,
    y_prob,
    thresholds: Iterable[float],
) -> tuple[float, float, float]:
    """
    Select threshold that maximizes recall.
    Tie-breakers: higher F1, then lower threshold.
    Returns (best_threshold, best_recall, best_f1).
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    best_threshold = None
    best_recall = -1.0
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        recall = recall_score(y_true_arr, y_pred, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)

        if (
            recall > best_recall
            or (np.isclose(recall, best_recall) and f1 > best_f1)
            or (
                np.isclose(recall, best_recall)
                and np.isclose(f1, best_f1)
                and (best_threshold is None or threshold < best_threshold)
            )
        ):
            best_threshold = float(threshold)
            best_recall = float(recall)
            best_f1 = float(f1)

    return best_threshold, best_recall, best_f1
