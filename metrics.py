"""Evaluation metrics on KNOWN cells only (mask M = 1), as in the report.

P, Y, M are arrays of shape (n_samples, rows, cols). One sample = one forecast origin
at one horizon, so a block bootstrap over samples is a bootstrap over 16-day blocks.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.metrics import average_precision_score


def _known(P, Y, M):
    s = M.astype(bool)
    return P[s], Y[s].astype(int)


def prevalence(Y, M):
    return float(Y[M.astype(bool)].mean())


def pr_auc(P, Y, M):
    p, y = _known(P, Y, M)
    return float("nan") if y.sum() == 0 else float(average_precision_score(y, p))


def dilate(A):
    """3x3 neighbourhood maximum over the last two axes (same or adjacent cell)."""
    return maximum_filter(A, size=(1, 3, 3), mode="constant", cval=0)


def tolerant_pr_auc(P, Y, M):
    """A known cell counts as positive if a positive lies in the same or an adjacent cell."""
    return pr_auc(P, dilate(Y) * M, M)


def _top_flags(P, M, q):
    """Boolean grid flagging the top q% of KNOWN cells (pooled over samples) by risk."""
    s = M.astype(bool)
    k = max(1, int(np.ceil(q / 100.0 * s.sum())))
    order = np.argsort(-P[s], kind="stable")[:k]
    flags = np.zeros(P.shape, dtype=bool)
    flat = np.zeros(int(s.sum()), dtype=bool); flat[order] = True
    flags[s] = flat
    return flags


def recall_at_alert_area(P, Y, M, q):
    flags = _top_flags(P, M, q)
    pos = (Y.astype(bool) & M.astype(bool))
    return float("nan") if pos.sum() == 0 else float((flags & pos).sum() / pos.sum())


def tolerant_recall_at_alert_area(P, Y, M, q):
    """A positive is captured if a flagged cell lies in the same or an adjacent cell."""
    flags = dilate(_top_flags(P, M, q).astype(np.uint8)).astype(bool)
    pos = (Y.astype(bool) & M.astype(bool))
    return float("nan") if pos.sum() == 0 else float((flags & pos).sum() / pos.sum())


def f1_at(P, Y, M, threshold):
    p, y = _known(P, Y, M)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def best_f1_threshold(P, Y, M):
    """Threshold maximising F1 (fit on validation, then frozen for test)."""
    p, y = _known(P, Y, M)
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 300)))
    scores = [f1_at(P, Y, M, t) for t in cand]
    i = int(np.argmax(scores))
    return float(cand[i]), float(scores[i])


def brier(P, Y, M):
    p, y = _known(P, Y, M)
    return float(np.mean((p - y) ** 2))


def reliability(P, Y, M, bins=10):
    p, y = _known(P, Y, M)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]


def block_bootstrap(metric, P, Y, M, n_boot=1000, seed=0):
    """95% interval of `metric` resampling forecast origins (16-day blocks) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(P); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(P[idx], Y[idx], M[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summary(P, Y, M, qs=(5, 10, 20), n_boot=1000):
    lo, hi = block_bootstrap(pr_auc, P, Y, M, n_boot)
    out = {"known_cells": int(M.sum()), "positives": int((Y.astype(bool) & M.astype(bool)).sum()),
           "prevalence": prevalence(Y, M), "pr_auc": pr_auc(P, Y, M),
           "pr_auc_lo": lo, "pr_auc_hi": hi, "tol_pr_auc": tolerant_pr_auc(P, Y, M),
           "brier": brier(P, Y, M)}
    for q in qs:
        out[f"recall@{q}%"] = recall_at_alert_area(P, Y, M, q)
        out[f"tol_recall@{q}%"] = tolerant_recall_at_alert_area(P, Y, M, q)
    return out
