"""Three-valued label builder for the AgriWatch locust module.

Label for (origin t0, horizon h, cell i):
  window W = [t0 + h, t0 + h + 16) days
  Y = 1, M = 1  if a qualifying record falls in cell i during W          (positive)
  Y = 0, M = 1  if a survey-confirmed absence falls in W and no positive  (negative)
  M = 0         otherwise (unknown: excluded from loss and metrics)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- grid
def latlon_to_cell(lat: float, lon: float) -> Optional[Tuple[int, int]]:
    """Cell (row, col) for a point; None if outside the study box.
    Row 0 is the northernmost row. Cells are half-open: [south, north), [west, east)."""
    if not (C.LAT_MIN <= lat < C.LAT_MAX and C.LON_MIN <= lon < C.LON_MAX):
        return None
    south_idx = int(np.floor((lat - C.LAT_MIN) / C.CELL))
    west_idx = int(np.floor((lon - C.LON_MIN) / C.CELL))
    return C.N_ROWS - 1 - south_idx, west_idx


def cell_from_corner(lat0: float, lon0: float) -> Optional[Tuple[int, int]]:
    """Cell for a 0.25-degree cell identified by its south-west corner (as in the
    pre-aggregated FAO tables). Corners must lie on the grid."""
    s = (lat0 - C.LAT_MIN) / C.CELL
    w = (lon0 - C.LON_MIN) / C.CELL
    if abs(s - round(s)) > 1e-6 or abs(w - round(w)) > 1e-6:
        raise ValueError(f"corner ({lat0}, {lon0}) is not on the 0.25-degree grid")
    s, w = int(round(s)), int(round(w))
    if not (0 <= s < C.N_ROWS and 0 <= w < C.N_COLS):
        return None
    return C.N_ROWS - 1 - s, w


# ----------------------------------------------------------------------- calendar
def composite_starts(year: int) -> List[date]:
    """MODIS-style 16-day composites: start on day-of-year 1, 17, ..., 353 (23 per year)."""
    jan1 = date(year, 1, 1)
    return [jan1 + timedelta(days=C.STEP_DAYS * k) for k in range(23)]


def composite_end(start: date) -> date:
    """Exclusive end of a composite; the last composite of a year is cut at 1 January."""
    nxt = start + timedelta(days=C.STEP_DAYS)
    jan1_next = date(start.year + 1, 1, 1)
    return min(nxt, jan1_next)


def forecast_origins(years: Iterable[int] = C.ORIGIN_YEARS) -> List[date]:
    """Origin t0 = end (exclusive) of each composite: the first day NOT covered by inputs."""
    return [composite_end(s) for y in years for s in composite_starts(y)]


def target_window(t0: date, h: int) -> Tuple[date, date]:
    """[start, end) of the target window for horizon h days."""
    ws = t0 + timedelta(days=h)
    return ws, ws + timedelta(days=C.TARGET_LEN)


def assign_block(ws: date, we: date,
                 blocks: Dict[str, Tuple[date, date]] = C.BLOCKS) -> str:
    """Block of a target window [ws, we). 'embargo' if it overlaps a block without
    being fully inside it; 'unused' if it overlaps none."""
    last_day = we - timedelta(days=1)
    overlapped = False
    for name, (bs, be) in blocks.items():
        if ws >= bs and last_day <= be:
            return name
        if ws <= be and last_day >= bs:
            overlapped = True
    return "embargo" if overlapped else "unused"


# -------------------------------------------------------------------------- events
class EventCubes:
    """Daily event-count cubes with prefix sums for fast window queries."""

    def __init__(self, n_days: int):
        shape = (n_days + 1, C.N_ROWS, C.N_COLS)
        self.n_days = n_days
        self.counts = {k: np.zeros((n_days, C.N_ROWS, C.N_COLS), dtype=np.int32)
                       for k in ("absence", *C.TARGETS)}
        self._prefix: Dict[str, np.ndarray] = {}
        self._shape = shape

    def add(self, name: str, day: date, cell: Tuple[int, int], n: int = 1):
        d = (day - C.EPOCH).days
        if 0 <= d < self.n_days:
            self.counts[name][d, cell[0], cell[1]] += n

    def finalize(self):
        for k, a in self.counts.items():
            p = np.zeros(self._shape, dtype=np.int32)
            np.cumsum(a, axis=0, out=p[1:])
            self._prefix[k] = p

    def window(self, name: str, ws: date, we: date) -> np.ndarray:
        """Event count per cell for days in [ws, we)."""
        a = max((ws - C.EPOCH).days, 0)
        b = min((we - C.EPOCH).days, self.n_days)
        if b <= a:
            return np.zeros((C.N_ROWS, C.N_COLS), dtype=np.int32)
        p = self._prefix[name]
        return p[b] - p[a]


def load_events(positives_csv: str, absences_csv: str) -> EventCubes:
    """Load the cell-date FAO tables (study box, 0.25-degree corners) into cubes.

    positives: category, lat0, lon0, country_name, start_date, n, area_ha
    absences : lat0, lon0, start_date, n_absence   (FAO 'NO LOCUST' records)
    """
    cubes = EventCubes((C.CALENDAR_END - C.EPOCH).days)
    pos = pd.read_csv(positives_csv, parse_dates=["start_date"])
    for cat, lat0, lon0, day, n in zip(pos.category, pos.lat0, pos.lon0,
                                       pos.start_date.dt.date, pos.n):
        cell = cell_from_corner(lat0, lon0)
        if cell is None:
            continue
        for tname, cats in C.TARGETS.items():
            if cat in cats:
                cubes.add(tname, day, cell, int(n))
    ab = pd.read_csv(absences_csv, parse_dates=["start_date"])
    for lat0, lon0, day, n in zip(ab.lat0, ab.lon0, ab.start_date.dt.date, ab.n_absence):
        cell = cell_from_corner(lat0, lon0)
        if cell is not None:
            cubes.add("absence", day, cell, int(n))
    cubes.finalize()
    return cubes


# -------------------------------------------------------------------------- labels
def build_labels(cubes: EventCubes, origins: List[date], target: str,
                 region_mask: Optional[np.ndarray] = None):
    """Return Y, M of shape (n_origins, n_horizons, rows, cols) as uint8.

    Cells with a qualifying record are positive; cells with an absence record and no
    qualifying record are negative; everything else is unknown (M = 0). Under the
    'breeding' target, a cell that has only adult/swarm records and no absence record
    is unknown, not negative.
    """
    if region_mask is None:
        region_mask = np.ones((C.N_ROWS, C.N_COLS), dtype=bool)
    O, H = len(origins), len(C.HORIZONS)
    Y = np.zeros((O, H, C.N_ROWS, C.N_COLS), dtype=np.uint8)
    M = np.zeros_like(Y)
    for i, t0 in enumerate(origins):
        for j, h in enumerate(C.HORIZONS):
            ws, we = target_window(t0, h)
            pos = cubes.window(target, ws, we) > 0
            neg = (cubes.window("absence", ws, we) > 0) & ~pos
            Y[i, j] = pos
            M[i, j] = (pos | neg) & region_mask
            Y[i, j] &= M[i, j]
    return Y, M


def block_table(origins: List[date]) -> np.ndarray:
    """Block name for every (origin, horizon) sample, shape (n_origins, n_horizons)."""
    out = np.empty((len(origins), len(C.HORIZONS)), dtype=object)
    for i, t0 in enumerate(origins):
        for j, h in enumerate(C.HORIZONS):
            out[i, j] = assign_block(*target_window(t0, h))
    return out
