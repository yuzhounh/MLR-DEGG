"""Metrics for the retained features of a fitted model."""

import numpy as np

from mlrdegg.experiment.support_eval import (anova_f_map, benjamini_hochberg, nogueira_stability,
                                             overlap_metrics, pairwise_jaccard, rank_metrics,
                                             retained_mask, sufficiency_necessity)


def test_overlap_metrics_perfect_and_random():
    R = np.zeros(100, bool)
    R[:10] = True
    m = overlap_metrics(R, R)
    assert m["precision"] == m["recall"] == m["dice"] == 1.0
    assert m["enrichment"] == 10.0 and m["p_hyper"] < 1e-10
    S = np.zeros(100, bool)
    S[50:60] = True  # disjoint from the reference
    m = overlap_metrics(S, R)
    assert m["precision"] == 0.0 and m["p_hyper"] == 1.0


def test_rank_metrics_follow_the_reference():
    score = np.arange(50.0)
    S = score >= 40
    m = rank_metrics(S, score, score)
    assert m["spearman"] > 0.999999 and m["topk_overlap"] == 1.0 and m["topk_chance"] == 0.2
    assert rank_metrics(score < 10, -score, score)["topk_overlap"] == 0.0  # lowest 10 vs top 10


def test_stability_identical_and_disjoint_sets():
    a = np.zeros(40, bool)
    a[:5] = True
    b = np.zeros(40, bool)
    b[20:25] = True
    assert pairwise_jaccard([a, a, a]) == 1.0 and nogueira_stability([a, a, a]) == 1.0
    assert pairwise_jaccard([a, b]) == 0.0 and nogueira_stability([a, b]) < 0


def test_retained_mask_ignores_numerical_zeros():
    assert retained_mask(np.array([0.0, 1e-14, 0.5, 2.0])).tolist() == [False, False, True, True]


def test_benjamini_hochberg_and_anova():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False]
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 40)
    X = rng.normal(size=(120, 4))
    X[:, 0] += y  # class effect on the first feature only
    F, d1, d2 = anova_f_map(X, y)
    assert (d1, d2) == (2, 117) and F[0] > 10 * F[1:].max() / 3


def test_sufficiency_necessity_detects_informative_features():
    rng = np.random.default_rng(1)
    n, p = 160, 60
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p))
    X[:, :6] += 1.5 * y[:, None]  # the first six features carry the signal
    S = np.zeros(p, bool)
    S[:6] = True
    r = sufficiency_necessity(X, y, np.arange(100), np.arange(100, n), S, n_random=30, C=0.1)
    assert r["acc_S"] > r["acc_random"] and r["p_suff"] < 0.1
    assert r["acc_without_S"] < r["acc_all"]
