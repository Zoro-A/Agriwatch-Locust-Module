"""Settings for the AgriWatch locust label builder.

Everything a supervisor decision could change (target definition, split dates)
lives here, so changing it never touches the logic in locust_labels.py.
"""
from datetime import date

# ---- Study box and grid (WGS84), as in the report: 28 x 24 cells of 0.25 deg
LAT_MIN, LAT_MAX = 24.0, 31.0
LON_MIN, LON_MAX = 68.0, 74.0
CELL = 0.25
N_ROWS = int(round((LAT_MAX - LAT_MIN) / CELL))   # 28 (row 0 = northernmost)
N_COLS = int(round((LON_MAX - LON_MIN) / CELL))   # 24 (col 0 = westernmost)

# ---- Time
EPOCH = date(2000, 1, 1)          # day 0 of the event cubes
CALENDAR_END = date(2023, 1, 1)   # exclusive end of the event cubes
STEP_DAYS = 16                    # MODIS 16-day composites (DOY 1, 17, ..., 353)
HORIZONS = (28, 42, 56)           # days from forecast origin to target window start
TARGET_LEN = 16                   # target window length in days
MAX_REACH = max(HORIZONS) + TARGET_LEN   # 72 days: how far a label looks ahead

# ---- Targets: which FAO categories count as a positive
TARGETS = {
    "any": ("ADULT", "HOPPER", "BAND", "SWARM"),   # presence at any life stage
    "breeding": ("HOPPER", "BAND"),                # local breeding evidence
}
PRIMARY_TARGET = "any"

# ---- Time blocks, by the dates of the TARGET WINDOW (inclusive both ends).
# A (origin, horizon) sample belongs to a block only if its whole target window
# lies inside it; windows straddling a boundary are embargoed (used nowhere).
BLOCKS = {
    "train": (date(2008, 1, 1), date(2019, 12, 31)),
    "val":   (date(2020, 1, 1), date(2020, 6, 30)),
    "test":  (date(2020, 7, 1), date(2020, 12, 31)),
}

# ---- Origins are generated for these years (composite calendar, see module)
ORIGIN_YEARS = range(2007, 2022)

# ---- Input features (anomalies relative to climatology)
INPUT_YEARS = range(2001, 2021)       # years extracted from Earth Engine
CLIM_YEARS = range(2001, 2020)        # climatology fitted on these years only (never 2020)
SEQ_LEN = 12                          # input steps per sample (about six months)
MIN_VALID_FRAC = 0.2                  # MODIS cell-step counts as observed above this valid-pixel share
Z_CLIP = 5.0                          # anomalies clipped to +/- this value (safety against tiny sigma)
CHANNELS = ("ndvi", "wi", "sm", "temp", "rain")   # plus one mask channel: 'modis_observed'
