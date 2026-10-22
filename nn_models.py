"""Neural models: M = ConvLSTM (three horizon heads) and B3 = per-cell LSTM (same heads).

Both read the same input sequence (12 steps x 6 channels) and are trained with a masked,
positive-weighted binary cross-entropy over KNOWN cells only. Training samples are origins; a
head contributes to the loss only where its (origin, horizon) label window is allowed by the
rolling-origin cut-off (otherwise its mask is zero).
"""
from __future__ import annotations
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
import rolling as R
from dataset import Samples
from sklearn.metrics import average_precision_score

NH = len(C.HORIZONS)
IN_CH = 6
VARIANTS = {"env": 6, "sight": 9, "sight_month": 11}   # input channels per variant


def build_input(S: Samples, k: int, variant: str = "env") -> np.ndarray:
    """(12, C, H, W) input for sample k. env: the 6 anomaly/mask channels; sight: + 3 recent-sightings
    channels (any stage, hopper/band, surveyed in the step); sight_month: + sin/cos of the step's
    calendar month (constant maps). Sightings of a step end exactly at the step's end (<= origin)."""
    x = S.window(k)
    if variant == "env":
        return x
    parts = [x, S.sight_window(k).astype(np.float32)]
    if variant == "sight_month":
        ends = S.step_end[S.win_start[k]:S.win_stop[k]]
        m = np.array([int(str(e)[5:7]) for e in ends])
        a = 2 * np.pi * (m - 1) / 12
        sc = np.stack([np.sin(a), np.cos(a)], 1).astype(np.float32)          # (12, 2)
        parts.append(np.broadcast_to(sc[:, :, None, None], (len(m), 2, C.N_ROWS, C.N_COLS)))
    return np.concatenate(parts, 1)


# ------------------------------------------------------------------ ConvLSTM
class ConvLSTMCell(nn.Module):
    def __init__(self, c_in: int, c_hid: int, k: int = 3):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMNet(nn.Module):
    """2 ConvLSTM layers (32, 64 channels, 3x3) and three 1x1 heads -> logits (B, 3, H, W)."""

    def __init__(self, c_in: int = IN_CH, hidden=(32, 64), dropout: float = 0.0):
        super().__init__()
        self.cells = nn.ModuleList([ConvLSTMCell(c_in, hidden[0]), ConvLSTMCell(hidden[0], hidden[1])])
        self.hidden = hidden
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(hidden[1], NH, 1)

    def forward(self, x):                      # x: (B, T, C, H, W)
        B, T, _, H, W = x.shape
        st = [(x.new_zeros(B, h, H, W), x.new_zeros(B, h, H, W)) for h in self.hidden]
        for t in range(T):
            inp = x[:, t]
            for li, cell in enumerate(self.cells):
                st[li] = cell(inp, st[li])
                inp = st[li][0]
        return self.head(self.drop(inp))


# ------------------------------------------------------------ per-cell LSTM
class CellLSTMNet(nn.Module):
    """One LSTM shared by all cells; each cell sees only its own 12-step history (no neighbours)."""

    def __init__(self, c_in: int = IN_CH, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, NH)

    def forward(self, x):                      # x: (B, T, C, H, W)
        B, T, Cc, H, W = x.shape
        seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, Cc)
        out, _ = self.lstm(seq)
        z = self.head(self.drop(out[:, -1]))   # (B*H*W, 3)
        return z.reshape(B, H, W, NH).permute(0, 3, 1, 2)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ data
class OriginData:
    """Per-origin tensors: inputs (T,6,H,W) and labels/masks for the three heads."""

    def __init__(self, S: Samples):
        self.S = S
        self.by_origin: Dict[int, Dict[int, int]] = {}
        for k in range(len(S.origin_idx)):
            self.by_origin.setdefault(int(S.origin_idx[k]), {})[int(S.hor_idx[k])] = k

    def x(self, oi: int, variant: str = "env") -> np.ndarray:
        k = next(iter(self.by_origin[oi].values()))
        return build_input(self.S, k, variant)

    def targets(self, oi: int, allowed: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Y, M of shape (3, H, W); M is zero for heads whose label is not allowed at this cut-off."""
        Y = np.zeros((NH, C.N_ROWS, C.N_COLS), np.float32)
        M = np.zeros((NH, C.N_ROWS, C.N_COLS), np.float32)
        for j, k in self.by_origin[oi].items():
            if k in allowed[j]:
                Y[j], M[j] = self.S.Y[k], self.S.M[k]
        return Y, M


def masked_bce(logits, Y, M, pos_weight):
    w = M * (1 + (pos_weight - 1) * Y)
    loss = F.binary_cross_entropy_with_logits(logits, Y, weight=w, reduction="sum")
    return loss / M.sum().clamp(min=1)


def _val_score(P: np.ndarray, Y: np.ndarray, M: np.ndarray) -> float:
    scores = []
    for j in range(NH):
        m = M[:, j].astype(bool)
        y = Y[:, j][m]
        if m.sum() and 0 < y.sum() < len(y):
            scores.append(average_precision_score(y, P[:, j][m]))
    return float(np.mean(scores)) if scores else 0.0


class NeuralRisk:
    """Fit on all heads at once for one rolling cut-off; predict any (origin, horizon)."""

    def __init__(self, kind: str = "convlstm", epochs: int = 30, patience: int = 6, lr: float = 2e-3,
                 weight_decay: float = 1e-4, batch: int = 8, val_frac: float = 0.2, seed: int = 0,
                 dropout: float = 0.1, max_pos_weight: float = 10.0, verbose: bool = False,
                 variant: str = "env"):
        self.kind, self.epochs, self.patience, self.lr = kind, epochs, patience, lr
        self.wd, self.batch, self.val_frac, self.seed = weight_decay, batch, val_frac, seed
        self.dropout, self.max_pw, self.verbose = dropout, max_pos_weight, verbose
        self.variant = variant
        self.net = None
        self.log: List[str] = []

    def _new(self):
        torch.manual_seed(self.seed)
        c = VARIANTS[self.variant]
        return (ConvLSTMNet(c_in=c, dropout=self.dropout) if self.kind == "convlstm"
                else CellLSTMNet(c_in=c, dropout=self.dropout))

    def fit(self, D: OriginData, cutoff: date, warm: "NeuralRisk" = None):
        S = D.S
        allowed = {j: set(R.train_indices(S, j, cutoff).tolist()) for j in range(NH)}
        origins = sorted({int(S.origin_idx[k]) for j in range(NH) for k in allowed[j]},
                         key=lambda oi: S.origins[oi])
        n_val = max(3, int(round(self.val_frac * len(origins))))
        va, tr = origins[-n_val:], origins[:-n_val]
        # embargo: a training head may not look at labels from windows overlapping the validation period
        v_start = min(S.tw_start[k] for oi in va for j, k in D.by_origin[oi].items() if k in allowed[j])
        allowed_tr = {j: {k for k in allowed[j] if S.tw_end[k] <= v_start} for j in range(NH)}
        Xtr = torch.tensor(np.stack([D.x(oi, self.variant) for oi in tr]))
        tgt = [D.targets(oi, allowed_tr) for oi in tr]
        Ytr = torch.tensor(np.stack([t[0] for t in tgt])); Mtr = torch.tensor(np.stack([t[1] for t in tgt]))
        Xva = torch.tensor(np.stack([D.x(oi, self.variant) for oi in va]))
        tv = [D.targets(oi, allowed) for oi in va]
        Yva = np.stack([t[0] for t in tv]); Mva = np.stack([t[1] for t in tv])
        n_pos = float((Ytr * Mtr).sum()); n_neg = float(Mtr.sum()) - n_pos
        pw = float(min(self.max_pw, max(1.0, n_neg / max(n_pos, 1.0))))
        net = self._new()
        if warm is not None and warm.net is not None:
            net.load_state_dict(warm.net.state_dict())
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        rng = np.random.RandomState(self.seed)
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.epochs):
            net.train()
            perm = rng.permutation(len(tr))
            tot = 0.0
            for b in range(0, len(perm), self.batch):
                ix = perm[b:b + self.batch]
                opt.zero_grad()
                loss = masked_bce(net(Xtr[ix]), Ytr[ix], Mtr[ix], pw)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(ix)
            net.eval()
            with torch.no_grad():
                Pv = torch.sigmoid(net(Xva)).numpy()
            sc = _val_score(Pv, Yva, Mva)
            self.log.append(f"ep{ep} loss {tot / len(tr):.4f} val-PRAUC {sc:.3f}")
            if self.verbose:
                print("   ", self.log[-1], flush=True)
            if sc > best:
                best, bad = sc, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net, self.best_val, self.epochs_run, self.n_train_origins = net, best, ep + 1, len(tr)
        return self

    def predict(self, D: OriginData, sample_ids: np.ndarray) -> np.ndarray:
        """Risk maps (n, H, W) for samples (each uses the head of its own horizon)."""
        S = D.S
        out = np.zeros((len(sample_ids), C.N_ROWS, C.N_COLS), np.float32)
        with torch.no_grad():
            for i, k in enumerate(sample_ids):
                P = torch.sigmoid(self.net(torch.tensor(build_input(S, k, self.variant))[None]))[0].numpy()
                out[i] = P[int(S.hor_idx[k])]
        return out
