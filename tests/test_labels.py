import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import config as C
import locust_labels as L

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ----------------------------------------------------------------- calendar
def test_23_composites_per_year_on_modis_days():
    for y in (2019, 2020):
        s = L.composite_starts(y)
        assert len(s) == 23
        assert [(d - date(y, 1, 1)).days + 1 for d in s] == list(range(1, 354, 16))


def test_last_composite_is_cut_at_new_year():
    assert L.composite_end(L.composite_starts(2019)[-1]) == date(2020, 1, 1)
    assert (date(2020, 1, 1) - L.composite_starts(2019)[-1]).days == 13   # non-leap
    assert (date(2021, 1, 1) - L.composite_starts(2020)[-1]).days == 14   # leap


def test_origins_unique_increasing_and_spacing():
    o = L.forecast_origins()
    assert o == sorted(set(o))
    gaps = {(b - a).days for a, b in zip(o, o[1:])}
    assert gaps <= {13, 14, 16, 3}   # 16 normally; short composite at year end
    assert max(gaps) == 16


def test_target_window_arithmetic():
    ws, we = L.target_window(date(2019, 7, 1), 28)
    assert (ws, we) == (date(2019, 7, 29), date(2019, 8, 14))
    ws, we = L.target_window(date(2019, 7, 1), 56)
    assert (we - ws).days == C.TARGET_LEN and (ws - date(2019, 7, 1)).days == 56
    assert C.MAX_REACH == 72


# --------------------------------------------------------------------- grid
def test_grid_shape():
    assert (C.N_ROWS, C.N_COLS) == (28, 24)


@pytest.mark.parametrize("lat,lon,cell", [
    (30.9, 68.1, (0, 0)), (24.0, 68.0, (27, 0)), (24.1, 73.9, (27, 23)),
    (30.99, 73.99, (0, 23)), (27.5, 71.0, (13 - 0, 12)),
])
def test_latlon_to_cell(lat, lon, cell):
    r, c = L.latlon_to_cell(lat, lon)
    exp_r = C.N_ROWS - 1 - int((lat - C.LAT_MIN) // C.CELL)
    assert (r, c) == (exp_r, int((lon - C.LON_MIN) // C.CELL))
    if cell != (13, 12):
        assert (r, c) == cell


@pytest.mark.parametrize("lat,lon", [(31.0, 70), (23.99, 70), (27, 74.0), (27, 67.99)])
def test_outside_box_is_none(lat, lon):
    assert L.latlon_to_cell(lat, lon) is None


def test_corner_and_point_agree():
    rng = np.random.default_rng(0)
    for _ in range(500):
        lat = rng.uniform(24, 31 - 1e-6); lon = rng.uniform(68, 74 - 1e-6)
        lat0 = np.floor(lat / 0.25) * 0.25; lon0 = np.floor(lon / 0.25) * 0.25
        assert L.latlon_to_cell(lat, lon) == L.cell_from_corner(lat0, lon0)


def test_off_grid_corner_raises():
    with pytest.raises(ValueError):
        L.cell_from_corner(27.1, 70.0)


# ------------------------------------------------------------------ blocks
def test_assign_block_basic():
    d = date
    assert L.assign_block(d(2015, 1, 1), d(2015, 1, 17)) == "train"
    assert L.assign_block(d(2019, 12, 16), d(2020, 1, 1)) == "train"      # last day Dec 31
    assert L.assign_block(d(2019, 12, 17), d(2020, 1, 2)) == "embargo"    # straddles
    assert L.assign_block(d(2020, 1, 1), d(2020, 1, 17)) == "val"
    assert L.assign_block(d(2020, 6, 15), d(2020, 7, 1)) == "val"
    assert L.assign_block(d(2020, 6, 16), d(2020, 7, 2)) == "embargo"
    assert L.assign_block(d(2020, 7, 1), d(2020, 7, 17)) == "test"
    assert L.assign_block(d(2007, 1, 1), d(2007, 1, 17)) == "unused"
    assert L.assign_block(d(2021, 3, 1), d(2021, 3, 17)) == "unused"
    assert L.assign_block(d(2007, 12, 25), d(2008, 1, 10)) == "embargo"


def test_blocks_are_disjoint_and_ordered():
    spans = sorted(C.BLOCKS.values())
    for (_, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 < s2


def test_no_label_leakage_between_blocks():
    """No training label window reaches validation/test; none of val reaches test."""
    origins = L.forecast_origins()
    tr_end = C.BLOCKS["train"][1]; va = C.BLOCKS["val"]; te = C.BLOCKS["test"]
    seen = {"train": 0, "val": 0, "test": 0, "embargo": 0}
    for t0 in origins:
        for h in C.HORIZONS:
            ws, we = L.target_window(t0, h)
            b = L.assign_block(ws, we)
            seen[b] = seen.get(b, 0) + 1
            last = we - timedelta(days=1)
            if b == "train":
                assert last <= tr_end
                assert t0 <= tr_end          # inputs end inside the training period
            elif b == "val":
                assert ws >= va[0] and last <= va[1]
            elif b == "test":
                assert ws >= te[0] and last <= te[1]
    assert seen["train"] > 0 and seen["val"] > 0 and seen["test"] > 0 and seen["embargo"] > 0


def test_embargo_removes_everything_straddling():
    for t0 in L.forecast_origins():
        for h in C.HORIZONS:
            ws, we = L.target_window(t0, h)
            if L.assign_block(ws, we) == "embargo":
                last = we - timedelta(days=1)
                ins = [n for n, (bs, be) in C.BLOCKS.items() if ws >= bs and last <= be]
                assert ins == []


# ------------------------------------------------------------------- labels
def _cubes(events=(), absences=()):
    """events: (target-cube-name(s), day, cell); absences: (day, cell)."""
    cu = L.EventCubes((C.CALENDAR_END - C.EPOCH).days)
    for names, day, cell in events:
        for nm in names:
            cu.add(nm, day, cell)
    for day, cell in absences:
        cu.add("absence", day, cell)
    cu.finalize()
    return cu


T0 = date(2019, 8, 1)            # origin
WS, WE = L.target_window(T0, 28)  # horizon index 0
A, B, Cc, D = (5, 5), (6, 6), (7, 7), (8, 8)


def test_window_edges_are_half_open():
    ev = [(("any",), WS - timedelta(days=1), A),      # before window
          (("any",), WS, B),                          # first day: inside
          (("any",), WE - timedelta(days=1), Cc),     # last day: inside
          (("any",), WE, D)]                          # end is exclusive: outside
    Y, M = L.build_labels(_cubes(ev), [T0], "any")
    y = Y[0, 0]
    assert (y[A], y[B], y[Cc], y[D]) == (0, 1, 1, 0)
    assert M[0, 0][A] == 0 and M[0, 0][D] == 0     # outside window: unknown


def test_positive_overrides_absence_and_absence_makes_negative():
    d = WS + timedelta(days=3)
    cu = _cubes(events=[(("any",), d, A)], absences=[(d, A), (d, B)])
    Y, M = L.build_labels(cu, [T0], "any")
    assert (Y[0, 0][A], M[0, 0][A]) == (1, 1)       # positive wins
    assert (Y[0, 0][B], M[0, 0][B]) == (0, 1)       # absence only -> negative
    assert (Y[0, 0][Cc], M[0, 0][Cc]) == (0, 0)     # nothing -> unknown


def test_breeding_target_ignores_adult_only_cells():
    d = WS + timedelta(days=3)
    # cell A: adult only; cell B: adult + absence; cell C: hopper
    ev = [(("any",), d, A), (("any",), d, B), (("any", "breeding"), d, Cc)]
    cu = _cubes(ev, absences=[(d, B)])
    Yb, Mb = L.build_labels(cu, [T0], "breeding")
    assert (Yb[0, 0][A], Mb[0, 0][A]) == (0, 0)     # adult-only: unknown for breeding
    assert (Yb[0, 0][B], Mb[0, 0][B]) == (0, 1)     # surveyed absent of hoppers: negative
    assert (Yb[0, 0][Cc], Mb[0, 0][Cc]) == (1, 1)
    Ya, Ma = L.build_labels(cu, [T0], "any")
    assert (Ya[0, 0][A], Ma[0, 0][A]) == (1, 1)


def test_horizons_use_different_windows():
    d = T0 + timedelta(days=45)      # inside the h=42 window [T0+42, T0+58), not h=28
    cu = _cubes([(("any",), d, A)])
    Y, M = L.build_labels(cu, [T0], "any")
    assert Y[0, 0][A] == 0 and Y[0, 1][A] == 1 and Y[0, 2][A] == 0


def test_region_mask_zeroes_labels():
    d = WS + timedelta(days=1)
    mask = np.ones((C.N_ROWS, C.N_COLS), dtype=bool); mask[A] = False
    Y, M = L.build_labels(_cubes([(("any",), d, A)], [(d, B)]), [T0], "any", mask)
    assert M[0, 0][A] == 0 and Y[0, 0][A] == 0 and M[0, 0][B] == 1


def test_label_invariants():
    d = WS + timedelta(days=2)
    cu = _cubes([(("any",), d, A)], [(d, B), (d, A)])
    Y, M = L.build_labels(cu, [T0, T0 + timedelta(days=16)], "any")
    assert set(np.unique(Y)) <= {0, 1} and set(np.unique(M)) <= {0, 1}
    assert (Y <= M).all()


# ---------------------------------------------------------------- real data
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "positives_box_celldate.csv")),
    reason="FAO tables not present")


@needs_data
def test_real_event_totals_match_csv():
    pcsv = os.path.join(DATA, "positives_box_celldate.csv")
    acsv = os.path.join(DATA, "absences_box_celldate.csv")
    cu = L.load_events(pcsv, acsv)
    p = pd.read_csv(pcsv, parse_dates=["start_date"])
    p = p[(p.start_date >= "2000-01-01") & (p.start_date < "2023-01-01")]
    a = pd.read_csv(acsv, parse_dates=["start_date"])
    a = a[(a.start_date >= "2000-01-01") & (a.start_date < "2023-01-01")]
    assert cu.counts["any"].sum() == p.n.sum()
    assert cu.counts["breeding"].sum() == p[p.category.isin(["HOPPER", "BAND"])].n.sum()
    assert cu.counts["absence"].sum() == a.n_absence.sum()


@needs_data
def test_real_record_lands_in_expected_cell_and_window():
    pcsv = os.path.join(DATA, "positives_box_celldate.csv")
    cu = L.load_events(pcsv, os.path.join(DATA, "absences_box_celldate.csv"))
    p = pd.read_csv(pcsv, parse_dates=["start_date"])
    r = p[(p.category == "HOPPER") & (p.start_date.dt.year == 2019)].iloc[100]
    cell = L.cell_from_corner(r.lat0, r.lon0)
    day = r.start_date.date()
    t0 = day - timedelta(days=42 + 5)         # day is 5 days into the h=42 window
    Y, M = L.build_labels(cu, [t0], "breeding")
    assert (Y[0, 1][cell], M[0, 1][cell]) == (1, 1)
