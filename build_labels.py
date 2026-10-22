"""Build label tensors from the downloaded FAO tables and write a summary."""
import os, sys
import numpy as np
import pandas as pd
import config as C
import locust_labels as L

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

cubes = L.load_events(os.path.join(DATA, "positives_box_celldate.csv"),
                      os.path.join(DATA, "absences_box_celldate.csv"))
origins = L.forecast_origins()
blocks = L.block_table(origins)

arrays = {"origins": np.array([o.isoformat() for o in origins]),
          "horizons": np.array(C.HORIZONS), "block": blocks.astype(str)}
rows = []
for target in C.TARGETS:
    Y, M = L.build_labels(cubes, origins, target)
    arrays[f"Y_{target}"], arrays[f"M_{target}"] = Y, M
    for b in list(C.BLOCKS) + ["embargo"]:
        for j, h in enumerate(C.HORIZONS):
            sel = blocks[:, j] == b
            if not sel.any():
                continue
            y, m = Y[sel, j], M[sel, j]
            rows.append(dict(target=target, block=b, horizon=h, origins=int(sel.sum()),
                             positive=int(y.sum()), negative=int((m & (1 - y)).sum()),
                             known_frac=round(float(m.mean()), 4)))
np.savez_compressed(os.path.join(OUT, "labels.npz"), **arrays)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "label_summary.csv"), index=False)
pd.set_option("display.width", 200)
print(summ.to_string(index=False))
