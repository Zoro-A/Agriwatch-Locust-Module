"""Rolling-origin evaluation.

For an evaluation forecast (origin t0, horizon h) a model may only use labels that were fully
observed by t0, i.e. training samples whose target window ENDED on or before t0. Models are
refitted every `group` evaluation origins (the first origin of the group sets the data cut-off),
which is conservative: later origins in a group do not get fresher training data.
"""
from __future__ import annotations
from datetime import date
from typing import Dict, List, Tuple

import numpy as np

import config as C
from dataset import Samples

TRAIN_START = date(2008, 1, 1)
EVAL_START, EVAL_END = date(2019, 1, 1), date(2020, 12, 31)


def eval_indices(S: Samples, j: int, start=EVAL_START, end=EVAL_END) -> np.ndarray:
    """Evaluation samples of horizon index j, ordered by origin."""
    ok = [k for k in np.where(S.hor_idx == j)[0]
          if S.tw_start[k] >= start and S.tw_end[k] <= date.fromordinal(end.toordinal() + 1)]
    return np.array(sorted(ok, key=lambda k: S.origin_date[k]))


def make_groups(eval_idx: np.ndarray, size: int) -> List[np.ndarray]:
    return [eval_idx[i:i + size] for i in range(0, len(eval_idx), size)]


def train_indices(S: Samples, j: int, cutoff: date) -> np.ndarray:
    """Samples of horizon j whose whole target window lies in [TRAIN_START, cutoff]."""
    return np.array([k for k in np.where(S.hor_idx == j)[0]
                     if S.tw_start[k] >= TRAIN_START and S.tw_end[k] <= cutoff])


class _Constant:
    def __init__(self, p): self.p = p
    def predict(self, S, idx): return np.full((len(idx), C.N_ROWS, C.N_COLS), self.p)


def rolling_predictions(S: Samples, j: int, factories: Dict[str, callable], group: int = 6,
                        verbose: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    ev = eval_indices(S, j)
    preds = {name: np.zeros((len(ev), C.N_ROWS, C.N_COLS)) for name in factories}
    pos = 0
    for g in make_groups(ev, group):
        cutoff = S.origin_date[g[0]]
        tr = train_indices(S, j, cutoff)
        n_pos, n_known = int(S.Y[tr].sum()), int(S.M[tr].sum())
        for name, make in factories.items():
            if n_pos < 5:
                model = _Constant(n_pos / max(n_known, 1))
            else:
                model = make().fit(S, tr)
            preds[name][pos:pos + len(g)] = model.predict(S, g)
        if verbose:
            print(f"  h={C.HORIZONS[j]} cutoff {cutoff} train {len(tr)} samples ({n_pos} pos) "
                  f"-> eval {len(g)}", flush=True)
        pos += len(g)
    return ev, preds
