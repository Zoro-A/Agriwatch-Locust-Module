from datetime import date

import numpy as np
import pandas as pd
import pytest

import config as C
import features as F


def _synthetic(years=(2001, 2002, 2003, 2004, 2005), seed=0, shift2005=0.0):
    """Small synthetic MODIS + ERA5-Land tables for the full grid, 23 steps per year."""
    rng = np.random.default_rng(seed)
    rows_m, rows_e = [], []
    rr, cc = np.meshgrid(range(C.N_ROWS), range(C.N_COLS), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    for y in years:
        for k in range(23):
            s = date(y, 1, 1) + pd.Timedelta(days=16 * k)
            n = len(rr)
            base = dict(step_start=s.isoformat(), step_end="x", row=rr, col=cc, lat0=0.0, lon0=0.0)
            add = shift2005 if y == 2005 else 0.0
            rows_m.append(pd.DataFrame({**base, "ndvi": 0.3 + 0.1 * np.sin(k) + rng.normal(0, .05, n) + add,
                                        "wi": 0.2 + rng.normal(0, .05, n) + add,
                                        "valid_frac": np.ones(n)}))
            rows_e.append(pd.DataFrame({**base, "n_days": 16,
                                        "t2m_c": 25 + 8 * np.sin(k / 3.6) + rng.normal(0, 1, n) + add,
                                        "precip_mm": rng.gamma(0.5, 5, n),
                                        "sm_l1": 0.15 + rng.normal(0, .02, n) + add * 0.01}))
    return pd.concat(rows_m, ignore_index=True), pd.concat(rows_e, ignore_index=True)


def test_shapes_and_alignment():
    m, e = _synthetic()
    out = F.build_features(m, e, clim_years=range(2001, 2005))
    assert out["X"].shape == (5 * 23, 6, C.N_ROWS, C.N_COLS)
    assert len(out["step_end"]) == 115
    assert out["step_end"][22] == "2002-01-01"        # last composite of 2001 ends 1 Jan


def test_climatology_never_uses_years_outside_clim_years():
    """Changing 2005 (outside CLIM_YEARS) must not change mu, sigma or any earlier anomaly."""
    m, e = _synthetic()
    m2, e2 = _synthetic(shift2005=3.0)
    a = F.build_features(m, e, clim_years=range(2001, 2005))
    b = F.build_features(m2, e2, clim_years=range(2001, 2005))
    for ch in C.CHANNELS:
        assert np.allclose(a["mu"][ch], b["mu"][ch], equal_nan=True)
        assert np.allclose(a["sigma"][ch], b["sigma"][ch], equal_nan=True)
    n_before_2005 = 4 * 23
    assert np.array_equal(a["X"][:n_before_2005], b["X"][:n_before_2005])
    assert not np.array_equal(a["X"][n_before_2005:], b["X"][n_before_2005:])   # 2005 does change


def test_anomalies_centred_on_clim_years():
    m, e = _synthetic()
    out = F.build_features(m, e, clim_years=range(2001, 2005))
    X = out["X"]; clim = out["year"] < 2005
    for ci in (2, 3):                                   # sm, temp: always observed
        assert abs(X[clim, ci].mean()) < 0.05
        assert 0.5 < X[clim, ci].std() < 1.1


def test_missing_modis_becomes_zero_with_mask():
    m, e = _synthetic()
    # make one cell-step unobserved two ways: NaN ndvi, and low valid fraction
    idx = m.index[(m.step_start == "2003-03-06") & (m.row == 4) & (m.col == 7)][0]
    m.loc[idx, "ndvi"] = np.nan
    idx2 = m.index[(m.step_start == "2003-03-06") & (m.row == 5) & (m.col == 8)][0]
    m.loc[idx2, "valid_frac"] = 0.05
    out = F.build_features(m, e, clim_years=range(2001, 2005))
    t = list(out["step_start"]).index("2003-03-06")
    X = out["X"]
    for (r, c) in ((4, 7), (5, 8)):
        assert X[t, 5, r, c] == 0                        # mask: unobserved
        assert X[t, 0, r, c] == 0 and X[t, 1, r, c] == 0  # filled with zero anomaly
    assert X[t, 5, 0, 0] == 1


def test_clipping_and_no_nans():
    m, e = _synthetic()
    out = F.build_features(m, e, clim_years=range(2001, 2005))
    assert np.isfinite(out["X"]).all()
    assert np.abs(out["X"][:, :5]).max() <= C.Z_CLIP + 1e-6


def test_mismatched_steps_rejected():
    m, e = _synthetic()
    with pytest.raises(ValueError):
        F.build_features(m, e[e.step_start != "2002-02-02"], clim_years=range(2001, 2005))


def test_input_window_ends_at_origin():
    ends = np.array([f"2019-{mo:02d}-01" for mo in range(1, 13)] + ["2020-01-01"])
    sl = F.input_window(ends, date(2019, 12, 1), seq_len=5)
    assert (sl.start, sl.stop) == (7, 12) and list(ends[sl])[-1] == "2019-12-01"
    assert F.input_window(ends, date(2019, 3, 1), seq_len=5) is None     # not enough history
    assert F.input_window(ends, date(2019, 3, 5), seq_len=5) is None     # not a step end
