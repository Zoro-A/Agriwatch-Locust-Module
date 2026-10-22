"""Assemble (origin, horizon) samples from the label and feature tensors."""
from __future__ import annotations
import os
from datetime import date
from typing import Dict, List

import numpy as np

import config as C
import features as F
import locust_labels as L

HERE = os.path.dirname(__file__)


class Samples:
    """Index of usable samples; inputs are sliced lazily from the feature tensor."""

    def __init__(self, target: str = C.PRIMARY_TARGET, out_dir: str = os.path.join(HERE, "output"),
                 include_all: bool = False):
        lab = np.load(os.path.join(out_dir, "labels.npz"), allow_pickle=True)
        fea = np.load(os.path.join(out_dir, "features.npz"), allow_pickle=True)
        self.X = fea["X"]                                  # (T, 6, 28, 24)
        self.step_end = fea["step_end"]
        sp = os.path.join(out_dir, "sightings.npz")
        self.sight = np.load(sp)["S"] if os.path.exists(sp) else None   # (T, 3, rows, cols)
        self.origins: List[date] = [date.fromisoformat(s) for s in lab["origins"]]
        Y, M, blocks = lab[f"Y_{target}"], lab[f"M_{target}"], lab["block"]
        rows = []
        for i, o in enumerate(self.origins):
            sl = F.input_window(self.step_end, o)
            if sl is None:
                continue
            for j, h in enumerate(C.HORIZONS):
                b = str(blocks[i, j])
                if include_all or b in C.BLOCKS:   # include_all: rolling-origin protocol ignores fixed blocks
                    rows.append((i, j, b, sl.start, sl.stop))
        self.origin_idx = np.array([r[0] for r in rows])
        self.hor_idx = np.array([r[1] for r in rows])
        self.block = np.array([r[2] for r in rows])
        self.win_start = np.array([r[3] for r in rows])
        self.win_stop = np.array([r[4] for r in rows])
        self.Y = np.stack([Y[r[0], r[1]] for r in rows]).astype(np.uint8)
        self.M = np.stack([M[r[0], r[1]] for r in rows]).astype(np.uint8)
        tw = [L.target_window(self.origins[r[0]], C.HORIZONS[r[1]]) for r in rows]
        self.tw_start = np.array([w[0] for w in tw])        # first day of the target window
        self.tw_end = np.array([w[1] for w in tw])          # exclusive end of the target window
        self.origin_date = np.array([self.origins[r[0]] for r in rows])
        self.target_month = np.array([
            L.target_window(self.origins[r[0]], C.HORIZONS[r[1]])[0].month for r in rows])

    def select(self, block: str, horizon_idx: int) -> np.ndarray:
        return np.where((self.block == block) & (self.hor_idx == horizon_idx))[0]

    def sight_window(self, k: int) -> np.ndarray:
        """Recent-sightings history for sample k: (SEQ_LEN, 3, 28, 24)."""
        return self.sight[self.win_start[k]:self.win_stop[k]]

    def window(self, k: int) -> np.ndarray:
        """Input sequence for sample k: (SEQ_LEN, 6, 28, 24)."""
        return self.X[self.win_start[k]:self.win_stop[k]]
