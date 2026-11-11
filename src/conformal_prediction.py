"""
Split-Conformal Prediction Engine.
Provides distribution-free, finite-sample statistical coverage guarantees
for classification and regression tasks without distributional assumptions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ConformalPredictor:
    """
    Split-Conformal Prediction Engine.
    Guarantees that for any target miscoverage rate alpha in (0, 1),
    P(Y in C(X)) >= 1 - alpha on exchangeable data.
    """

    def __init__(self, alpha: float = 0.10):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Miscoverage rate alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.classes: Optional[np.ndarray] = None
        self.task_type: Optional[str] = None
        self.n_calib: int = 0

    def fit_classifier(
        self,
        calib_probs: np.ndarray,
        calib_labels: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calibrate classifier on held-out calibration split.
        Non-conformity score: s_i = 1 - P(Y = y_i | X = x_i).
        """
        calib_probs = np.asarray(calib_probs)
        calib_labels = np.asarray(calib_labels)
        self.n_calib = len(calib_labels)

        if len(calib_probs) != self.n_calib:
            raise ValueError("calib_probs and calib_labels must have the same length")

        if classes is None:
            self.classes = np.unique(calib_labels)
        else:
            self.classes = np.asarray(classes)

        # Map labels to column indices
        label_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        true_indices = np.array([label_to_idx[y] for y in calib_labels])

        # Extract true class probabilities
        row_indices = np.arange(self.n_calib)
        true_probs = calib_probs[row_indices, true_indices]

        # Non-conformity scores: 1 - P(true_class)
        scores = 1.0 - true_probs

        # Finite-sample adjusted conformal quantile
        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), self.n_calib)
        self.q_hat = float(np.sort(scores)[k - 1])
        self.task_type = "classification"
        return self.q_hat

    def predict_set_classifier(
        self, test_probs: np.ndarray
    ) -> List[List[Union[int, str, float]]]:
        """
        Generate conformal prediction sets for test probabilities.
        C(x) = {k : P(Y = k | x) >= 1 - q_hat}
        """
        if self.q_hat is None or self.classes is None:
            raise RuntimeError("Model must be calibrated via fit_classifier first")

        test_probs = np.asarray(test_probs)
        threshold = 1.0 - self.q_hat

        prediction_sets: List[List[Union[int, str, float]]] = []
        for probs in test_probs:
            included_indices = np.where(probs >= threshold)[0]
            # If threshold is strict and set is empty, include argmax class
            if len(included_indices) == 0:
                included_indices = np.array([np.argmax(probs)])
            prediction_sets.append([self.classes[i] for i in included_indices])

        return prediction_sets

    def evaluate_classification(
        self, test_probs: np.ndarray, test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate empirical coverage, average set size, and singleton efficiency.
        """
        test_labels = np.asarray(test_labels)
        pred_sets = self.predict_set_classifier(test_probs)

        covered = sum(1 for y, s in zip(test_labels, pred_sets) if y in s)
        set_sizes = [len(s) for s in pred_sets]

        coverage_rate = covered / len(test_labels) if len(test_labels) > 0 else 0.0
        avg_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0
        singleton_pct = (
            sum(1 for sz in set_sizes if sz == 1) / len(set_sizes) * 100.0
            if set_sizes
            else 0.0
        )
        ambiguous_pct = (
            sum(1 for sz in set_sizes if sz > 1) / len(set_sizes) * 100.0
            if set_sizes
            else 0.0
        )

        return {
            "target_coverage_pct": (1.0 - self.alpha) * 100.0,
            "empirical_coverage_pct": coverage_rate * 100.0,
            "conformal_quantile_qhat": self.q_hat if self.q_hat is not None else 0.0,
            "avg_prediction_set_size": round(avg_set_size, 3),
            "singleton_set_pct": round(singleton_pct, 2),
            "ambiguous_set_pct": round(ambiguous_pct, 2),
            "total_test_instances": len(test_labels),
        }

    def fit_regressor(
        self, calib_preds: np.ndarray, calib_y: np.ndarray
    ) -> float:
        """
        Calibrate regression model on held-out split.
        Non-conformity score: s_i = |y_i - mu_hat(x_i)|.
        """
        calib_preds = np.asarray(calib_preds).ravel()
        calib_y = np.asarray(calib_y).ravel()
        self.n_calib = len(calib_y)

        if len(calib_preds) != self.n_calib:
            raise ValueError("calib_preds and calib_y must have identical length")

        scores = np.abs(calib_y - calib_preds)
        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), self.n_calib)
        self.q_hat = float(np.sort(scores)[k - 1])
        self.task_type = "regression"
        return self.q_hat

    def predict_interval_regressor(
        self, test_preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return lower and upper conformal prediction intervals:
        [mu_hat(x) - q_hat, mu_hat(x) + q_hat].
        """
        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated via fit_regressor first")
        test_preds = np.asarray(test_preds).ravel()
        lower = test_preds - self.q_hat
        upper = test_preds + self.q_hat
        return lower, upper

    def evaluate_regression(
        self, test_preds: np.ndarray, test_y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression interval coverage and average band width.
        """
        test_preds = np.asarray(test_preds).ravel()
        test_y = np.asarray(test_y).ravel()
        lower, upper = self.predict_interval_regressor(test_preds)

        covered = np.sum((test_y >= lower) & (test_y <= upper))
        coverage_rate = float(covered / len(test_y)) if len(test_y) > 0 else 0.0
        avg_width = float(np.mean(upper - lower))

        return {
            "target_coverage_pct": (1.0 - self.alpha) * 100.0,
            "empirical_coverage_pct": round(coverage_rate * 100.0, 2),
            "margin_of_uncertainty_qhat": round(self.q_hat, 4) if self.q_hat is not None else 0.0,
            "avg_interval_width": round(avg_width, 4),
            "total_test_instances": len(test_y),
        }
