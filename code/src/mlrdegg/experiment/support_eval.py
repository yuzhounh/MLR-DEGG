"""Evaluation of the retained features (support) of a fitted model.

A retained set ``S`` is a boolean mask over the features (channels or voxels, optionally times).
It is judged on three kinds of evidence:

* agreement with a reference of task-related features ``R`` that is independent of the fit (the
  true support in simulations; a leave-one-subject-out group effect map on real data):
  :func:`overlap_metrics` for a binary reference and :func:`rank_metrics` for a continuous one;
* stability of ``S`` over resamples (:func:`pairwise_jaccard`, :func:`nogueira_stability`);
* decoding relevance of ``S`` against random sets of the same size
  (:func:`sufficiency_necessity`).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import hypergeom, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def retained_mask(importance: np.ndarray, rel_tol: float = 1e-10) -> np.ndarray:
    """Features with a nonzero importance (up to ``rel_tol`` of the largest one)."""
    imp = np.asarray(importance, float)
    return imp > rel_tol * max(imp.max(), 1e-300)


def overlap_metrics(S: np.ndarray, R: np.ndarray) -> dict:
    """Precision, recall, Dice, enrichment factor and hypergeometric p-value of ``S`` against ``R``.

    ``enrichment`` is the precision divided by the fraction of reference features among all
    features (1 means no better than a random set of the same size); ``p_hyper`` is the
    probability of an overlap at least as large for a random set of the same size."""
    S, R = np.asarray(S, bool).ravel(), np.asarray(R, bool).ravel()
    p, nS, nR, tp = S.size, int(S.sum()), int(R.sum()), int((S & R).sum())
    return {"n_selected": nS, "n_reference": nR, "reference_fraction": nR / p,
            "precision": tp / nS if nS else np.nan, "recall": tp / nR if nR else np.nan,
            "dice": 2 * tp / (nS + nR) if nS + nR else np.nan,
            "enrichment": (tp / nS) / (nR / p) if nS and nR else np.nan,
            "p_hyper": float(hypergeom.sf(tp - 1, p, nR, nS)) if nS and nR else np.nan}


def rank_metrics(S: np.ndarray, importance: np.ndarray, score: np.ndarray) -> dict:
    """Threshold-free agreement with a continuous reference ``score`` (larger = more task related).

    ``auc`` and ``spearman`` compare the importance with the reference over all features;
    ``topk_overlap`` is the fraction of ``S`` inside the ``|S|`` highest-scoring features (the
    chance level is ``|S| / p``), which avoids choosing a threshold for the reference."""
    S = np.asarray(S, bool).ravel()
    imp, score = np.asarray(importance, float).ravel(), np.asarray(score, float).ravel()
    out = {"spearman": np.nan, "topk_overlap": np.nan, "topk_chance": np.nan}
    if imp.std() > 0 and score.std() > 0:
        out["spearman"] = float(spearmanr(imp, score)[0])
    k = int(S.sum())
    if 0 < k < S.size:
        top = rankdata(-score, method="ordinal") <= k
        out["topk_overlap"] = float((S & top).sum() / k)
        out["topk_chance"] = k / S.size
    return out


def auc_against_reference(importance: np.ndarray, R: np.ndarray) -> float:
    """AUC of the importance for separating reference from non-reference features."""
    R = np.asarray(R, bool).ravel()
    imp = np.asarray(importance, float).ravel()
    if R.all() or not R.any() or imp.std() == 0:
        return np.nan
    return float(roc_auc_score(R, imp))


def pairwise_jaccard(masks: list[np.ndarray]) -> float:
    """Mean Jaccard index over all pairs of retained sets (pairs of empty sets are skipped)."""
    vals = []
    for a, b in itertools.combinations([np.asarray(m, bool).ravel() for m in masks], 2):
        union = (a | b).sum()
        if union:
            vals.append((a & b).sum() / union)
    return float(np.mean(vals)) if vals else np.nan


def nogueira_stability(masks: list[np.ndarray]) -> float:
    """Nogueira et al. (2018) stability of feature selection: 1 for identical sets, about 0 for
    sets no more similar than random sets of the same size, from ``M >= 2`` selections."""
    Z = np.stack([np.asarray(m, bool).ravel() for m in masks]).astype(float)
    M, p = Z.shape
    kbar = Z.sum(1).mean()
    if M < 2 or kbar == 0 or kbar == p:
        return np.nan
    freq = Z.mean(0)
    var = M / (M - 1) * freq * (1 - freq)
    return float(1.0 - var.mean() / (kbar / p * (1 - kbar / p)))


def _accuracy(X, y, tr, te, cols, C):
    if len(cols) == 0:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, tol=1e-5))
    clf.fit(X[np.ix_(tr, cols)], y[tr])
    return float(np.mean(clf.predict(X[np.ix_(te, cols)]) == y[te]))


def sufficiency_necessity(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                          S: np.ndarray, n_random: int = 50, C: float = 0.01, seed: int = 0) -> dict:
    """Decoding with only ``S`` (sufficiency) and without ``S`` (necessity) against random sets.

    An L2 logistic regression is fitted on the training samples and scored on the test samples.
    ``p_suff`` is the share of random sets of the same size that decode at least as well as ``S``;
    ``p_nec`` is the share of random sets whose removal hurts at least as much as removing ``S``.
    Both use the add-one correction, so the smallest possible value is ``1 / (n_random + 1)``.
    ``S`` must have been chosen without the test samples."""
    S = np.asarray(S, bool).ravel()
    idx = np.flatnonzero(S)
    rest = np.flatnonzero(~S)
    out = {"acc_all": _accuracy(X, y, train, test, np.arange(S.size), C),
           "acc_S": _accuracy(X, y, train, test, idx, C),
           "acc_without_S": _accuracy(X, y, train, test, rest, C)}
    rng = np.random.default_rng(seed)
    only, minus = [], []
    for _ in range(n_random):
        r = rng.permutation(S.size)[: len(idx)]
        mask = np.zeros(S.size, bool)
        mask[r] = True
        only.append(_accuracy(X, y, train, test, np.flatnonzero(mask), C))
        minus.append(_accuracy(X, y, train, test, np.flatnonzero(~mask), C))
    only, minus = np.asarray(only), np.asarray(minus)
    out.update(acc_random=float(np.nanmean(only)), acc_without_random=float(np.nanmean(minus)),
               p_suff=float((1 + np.sum(only >= out["acc_S"])) / (n_random + 1)),
               p_nec=float((1 + np.sum(minus <= out["acc_without_S"])) / (n_random + 1)))
    return out


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of the hypotheses rejected by the BH procedure at false discovery rate ``q``."""
    p = np.asarray(p, float).ravel()
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, p.size + 1) / p.size
    keep = np.zeros(p.size, bool)
    if passed.any():
        keep[order[: np.flatnonzero(passed).max() + 1]] = True
    return keep


def anova_f_map(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int, int]:
    """One-way ANOVA F statistic of every feature across the classes of ``y``, with its
    degrees of freedom ``(df_between, df_within)``.  ``X`` has shape ``(n, ...)``."""
    X = np.asarray(X, float)
    classes = np.unique(y)
    n, k = len(y), len(classes)
    grand = X.mean(0)
    ssb = sum((y == c).sum() * (X[y == c].mean(0) - grand) ** 2 for c in classes)
    ssw = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum(0) for c in classes)
    return (ssb / (k - 1)) / np.maximum(ssw / (n - k), 1e-300), k - 1, n - k
