"""Anomaly input tensors for the locust ConvLSTM.

x_c(i, t) -> z_c(i, t) = (x - mu_c(i, k)) / (sigma_c(i, k) + eps_c), k = step-of-year,
with mu and sigma fitted on CLIM_YEARS only. Missing values become 0 after a mask
channel records them (report, Section 'Data Sources and Input Channels').
"""
from __future__ import annotations
import glob, os, warnings
from datetime import date, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import config as C
import locust_labels as L


def load_inputs(folder: str):
    modis = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "locust_modis_*.csv")))])
    era = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "locust_era5land_*.csv")))])
    return modis, era


def _cube(df: pd.DataFrame, col: str, steps: pd.Index) -> np.ndarray:
    cube = np.full((len(steps), C.N_ROWS, C.N_COLS), np.nan, dtype=np.float64)
    cube[steps.get_indexer(df.step_start), df.row.values, df.col.values] = df[col].values
    return cube


def build_features(modis: pd.DataFrame, era: pd.DataFrame,
                   clim_years: Sequence[int] = C.CLIM_YEARS):
    steps = pd.Index(sorted(set(modis.step_start) | set(era.step_start)))
    if set(modis.step_start) != set(era.step_start):
        raise ValueError("MODIS and ERA5-Land step lists differ")
    starts = pd.to_datetime(steps)
    years = starts.year.values
    kidx = ((starts.dayofyear.values - 1) // C.STEP_DAYS).astype(int)      # step-of-year 0..22

    vf = _cube(modis, "valid_frac", steps)
    ndvi, wi = _cube(modis, "ndvi", steps), _cube(modis, "wi", steps)
    bad = ~np.isfinite(ndvi) | ~np.isfinite(wi) | ~(vf >= C.MIN_VALID_FRAC)
    ndvi[bad] = np.nan; wi[bad] = np.nan
    raw = {"ndvi": ndvi, "wi": wi,
           "sm": _cube(era, "sm_l1", steps),
           "temp": _cube(era, "t2m_c", steps),
           "rain": np.log1p(np.clip(_cube(era, "precip_mm", steps), 0, None))}

    in_clim = np.isin(years, list(clim_years))
    T = len(steps)
    X = np.zeros((T, len(C.CHANNELS) + 1, C.N_ROWS, C.N_COLS), dtype=np.float32)
    mu_all, sd_all = {}, {}
    observed = np.ones((T, C.N_ROWS, C.N_COLS), dtype=bool)
    for ci, name in enumerate(C.CHANNELS):
        a = raw[name]
        mu = np.full((23, C.N_ROWS, C.N_COLS), np.nan); sd = np.full_like(mu, np.nan)
        for k in range(23):
            sel = a[in_clim & (kidx == k)]
            if len(sel) == 0:
                continue
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cnt = np.isfinite(sel).sum(axis=0)
                m = np.nanmean(sel, axis=0); s = np.nanstd(sel, axis=0, ddof=1)
            m[cnt < 3] = np.nan; s[cnt < 3] = np.nan          # too few years: treat as unobserved
            mu[k], sd[k] = m, s
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eps = 0.1 * np.nanmedian(sd)
        z = (a - mu[kidx]) / (sd[kidx] + eps)
        z = np.clip(z, -C.Z_CLIP, C.Z_CLIP)
        fin = np.isfinite(z)
        if name in ("ndvi", "wi"):
            observed &= fin
        X[:, ci] = np.where(fin, z, 0.0)
        mu_all[name], sd_all[name] = mu, sd
    X[:, -1] = observed
    ends = np.array([L.composite_end(d.date()).isoformat() for d in starts])
    return {"X": X, "step_start": np.array(steps), "step_end": ends,
            "year": years, "k": kidx, "mu": mu_all, "sigma": sd_all}


def input_window(step_end: np.ndarray, origin: date, seq_len: int = C.SEQ_LEN) -> Optional[slice]:
    """Slice of the `seq_len` steps whose last step ends exactly at the forecast origin."""
    hit = np.where(step_end == origin.isoformat())[0]
    if len(hit) != 1:
        return None
    i = int(hit[0])
    return slice(i - seq_len + 1, i + 1) if i - seq_len + 1 >= 0 else None
