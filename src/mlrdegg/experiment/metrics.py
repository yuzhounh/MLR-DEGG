"""Classification metrics computed from predicted probabilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    pred = classes[np.argmax(proba, axis=1)]
    onehot = (y_true[:, None] == classes[None, :]).astype(float)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=classes)),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
    }


def _scaled_nll(logp: np.ndarray, idx: np.ndarray, tau: float) -> float:
    z = logp / tau
    z = z - z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(idx)), idx]))


def fit_temperature(proba: np.ndarray, idx: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """Temperature ``tau`` minimizing the log loss of ``softmax(log(proba) / tau)`` (temperature
    scaling, Guo et al., 2017); ``tau > 1`` softens overconfident predictions.  ``idx`` holds the
    column of the true class of every row."""
    logp = np.log(np.clip(proba, 1e-15, None))
    r = minimize_scalar(lambda s: _scaled_nll(logp, idx, np.exp(s)), bounds=np.log(bounds),
                        method="bounded")
    return float(np.exp(r.x))


def cross_fitted_temperature(probas: list[np.ndarray], idxs: list[np.ndarray]) -> list[tuple[float, float]]:
    """``(tau, scaled log loss)`` for every validation fold, with ``tau`` fitted on the predictions
    of the other folds (on the fold itself when there is only one), so the scaled log loss of a
    fold never uses its own labels."""
    out = []
    for i in range(len(probas)):
        others = [j for j in range(len(probas)) if j != i] or [i]
        tau = fit_temperature(np.concatenate([probas[j] for j in others]),
                              np.concatenate([idxs[j] for j in others]))
        out.append((tau, _scaled_nll(np.log(np.clip(probas[i], 1e-15, None)), idxs[i], tau)))
    return out


def selection_score(metrics: dict, name: str) -> float:
    """Larger is better.  ``neg_log_loss_ts`` is the temperature-scaled log loss (see
    :func:`cross_fitted_temperature`), which ranks grid points by how well they separate the
    classes regardless of over- or underconfidence."""
    if name == "neg_log_loss":
        return -metrics["log_loss"]
    if name == "neg_log_loss_ts":
        return -metrics["log_loss_ts"]
    if name == "neg_brier":
        return -metrics["brier"]
    return metrics[name]
