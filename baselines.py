"""Baseline ladder: B0 climatological prior, B1 logistic, B2 random forest, B2b LightGBM.

All models predict a risk map P (n, rows, cols) for the given samples and are fitted on
known cells of the TRAIN block only. No coordinates are used as features, so B0 is the
only model with a static spatial prior.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import config as C
from dataset import Samples

R, W = C.N_ROWS, C.N_COLS


# ------------------------------------------------------------------ B0 prior
class ClimatologicalPrior:
    """P(positive | cell, calendar month of the target window), from training labels,
    smoothed over the 3x3 neighbourhood and shrunk to the monthly average."""

    def __init__(self, shrink: float = 5.0):
        self.shrink = shrink

    def fit(self, S: Samples, idx: np.ndarray):
        pos = np.zeros((13, R, W)); known = np.zeros((13, R, W))
        for k in idx:
            m = S.target_month[k]
            pos[m] += S.Y[k]; known[m] += S.M[k]
        self.global_p = pos.sum() / max(known.sum(), 1)
        self.prior = np.zeros((13, R, W))
        for m in range(1, 13):
            ps = uniform_filter(pos[m], size=3, mode="constant") * 9
            ks = uniform_filter(known[m], size=3, mode="constant") * 9
            pm = pos[m].sum() / known[m].sum() if known[m].sum() > 0 else self.global_p
            self.prior[m] = (ps + self.shrink * pm) / (ks + self.shrink)
        return self

    def predict(self, S: Samples, idx: np.ndarray):
        return np.stack([self.prior[S.target_month[k]] for k in idx])


# ------------------------------------------------------- tabular features
def _month_enc(month):
    a = 2 * np.pi * (month - 1) / 12
    return np.sin(a), np.cos(a)


def _cell_features(S: Samples, k: int, kind: str) -> np.ndarray:
    """(R*W, n_features) feature matrix for one sample."""
    win = S.window(k)                       # (12, 6, R, W)
    last = win[-1]                          # (6, R, W)
    s, c = _month_enc(S.target_month[k])
    month = [np.full((R, W), s), np.full((R, W), c)]
    if kind == "last":                      # B1: current step only
        cols = list(last) + month
    else:                                   # B2: last 12 steps of the five anomaly channels
        cols = [win[t, ch] for t in range(win.shape[0]) for ch in range(5)] + [last[5]] + month
        if kind in ("spatial", "sight"):    # B2b: add neighbourhood means of the last step
            for size in (3, 7):
                cols += [uniform_filter(last[ch], size=size, mode="nearest") for ch in range(5)]
        if kind == "sight":                 # ablation: recent sightings (any, hopper/band, surveyed)
            sw = S.sight_window(k).astype(np.float32)           # (12, 3, R, W)
            cols += [sw[t, c] for t in range(sw.shape[0]) for c in range(3)]
            near = sw[-3:, 0].max(axis=0)                        # any record in the last 3 steps
            cols += [maximum_filter(near, size=3, mode="constant"),
                     maximum_filter(near, size=7, mode="constant")]
    return np.stack(cols, axis=-1).reshape(R * W, -1)


def _training_table(S, idx, kind):
    Xs, ys = [], []
    for k in idx:
        f = _cell_features(S, k, kind)
        m = S.M[k].reshape(-1).astype(bool)
        Xs.append(f[m]); ys.append(S.Y[k].reshape(-1)[m])
    return np.concatenate(Xs), np.concatenate(ys)


class TabularModel:
    def __init__(self, name: str, kind: str, make):
        self.name, self.kind, self.make = name, kind, make

    def fit(self, S: Samples, idx: np.ndarray):
        X, y = _training_table(S, idx, self.kind)
        self.scaler = StandardScaler().fit(X) if self.name == "B1 logistic" else None
        self.model = self.make()
        self.model.fit(self.scaler.transform(X) if self.scaler else X, y)
        return self

    def predict(self, S: Samples, idx: np.ndarray):
        out = []
        for k in idx:
            f = _cell_features(S, k, self.kind)
            if self.scaler:
                f = self.scaler.transform(f)
            out.append(self.model.predict_proba(f)[:, 1].reshape(R, W))
        return np.stack(out)


class Persistence:
    """Score = exponentially decayed record history of the cell itself (half per 16-day step)
    plus a smaller share from its 3x3 neighbourhood. No fitting; uses sightings only."""

    def __init__(self, channel: int, decay: float = 0.5, neigh: float = 0.2):
        self.channel, self.decay, self.neigh = channel, decay, neigh

    def fit(self, S, idx):
        return self

    def predict(self, S, idx):
        out = []
        for k in idx:
            h = S.sight_window(k)[:, self.channel].astype(np.float64)       # (12, R, W)
            w = self.decay ** np.arange(h.shape[0] - 1, -1, -1)
            own = np.tensordot(w, h, axes=(0, 0))
            nb = maximum_filter(own, size=3, mode="constant")
            out.append((own + self.neigh * nb) / (w.sum() * (1 + self.neigh)))
        return np.stack(out)


def make_sighting_models(target: str, seed: int = 0, rf_trees: int = 300):
    ch = 0 if target == "any" else 1
    return {
        "P persistence (sightings only)": Persistence(ch),
        "S2 random forest + sightings": TabularModel("S2 random forest + sightings", "sight",
                                                     lambda: RandomForestClassifier(
                                                         n_estimators=rf_trees, min_samples_leaf=5,
                                                         n_jobs=-1, random_state=seed)),
        "S2b LightGBM + sightings": TabularModel("S2b LightGBM + sightings", "sight",
                                                 lambda: lgb.LGBMClassifier(
                                                     n_estimators=300, learning_rate=0.05,
                                                     num_leaves=15, min_child_samples=20,
                                                     subsample=0.8, subsample_freq=1,
                                                     colsample_bytree=0.8, random_state=seed,
                                                     verbose=-1, n_jobs=2)),
    }


def make_models(seed: int = 0, rf_trees: int = 300):
    return {
        "B0 prior": ClimatologicalPrior(),
        "B1 logistic": TabularModel("B1 logistic", "last",
                                    lambda: LogisticRegression(max_iter=2000, C=1.0)),
        "B2 random forest": TabularModel("B2 random forest", "flat",
                                         lambda: RandomForestClassifier(
                                             n_estimators=rf_trees, min_samples_leaf=5, n_jobs=-1,
                                             random_state=seed)),
        "B2b LightGBM+neighbours": TabularModel("B2b LightGBM+neighbours", "spatial",
                                                lambda: lgb.LGBMClassifier(
                                                    n_estimators=300, learning_rate=0.05,
                                                    num_leaves=15, min_child_samples=20,
                                                    subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, random_state=seed,
                                                    verbose=-1, n_jobs=2)),
    }
