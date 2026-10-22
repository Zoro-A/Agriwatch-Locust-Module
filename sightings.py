"""Recent-sightings history per forecast step (used ONLY by the 'recent sightings' ablations).

For each 16-day input step: was a record of any life stage, a hopper/band record, or any survey
(record or confirmed absence) made in the cell during the step [start, end)? A step ends exactly
at the forecast origin, so no record dated on or after the origin is ever included.
"""
import os
from datetime import date
import numpy as np
import config as C
import locust_labels as L

HERE = os.path.dirname(__file__)


def build_step_sightings(cubes: L.EventCubes, step_start, step_end) -> np.ndarray:
    """(T, 3, rows, cols) uint8: channel 0 any-stage, 1 hopper/band, 2 surveyed."""
    out = np.zeros((len(step_start), 3, C.N_ROWS, C.N_COLS), dtype=np.uint8)
    for i, (s, e) in enumerate(zip(step_start, step_end)):
        s, e = date.fromisoformat(str(s)), date.fromisoformat(str(e))
        anyp = cubes.window("any", s, e) > 0
        br = cubes.window("breeding", s, e) > 0
        ab = cubes.window("absence", s, e) > 0
        out[i, 0], out[i, 1], out[i, 2] = anyp, br, anyp | ab
    return out


if __name__ == "__main__":
    fea = np.load(os.path.join(HERE, "output", "features.npz"), allow_pickle=True)
    cubes = L.load_events(os.path.join(HERE, "data", "positives_box_celldate.csv"),
                          os.path.join(HERE, "data", "absences_box_celldate.csv"))
    S = build_step_sightings(cubes, fea["step_start"], fea["step_end"])
    np.savez_compressed(os.path.join(HERE, "output", "sightings.npz"), S=S)
    print("sightings", S.shape, "share of cell-steps with any-stage record:", round(float(S[:, 0].mean()), 4))
