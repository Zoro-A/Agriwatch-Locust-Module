import os
import numpy as np
import features as F, locust_labels as L, config as C

HERE = os.path.dirname(__file__)
modis, era = F.load_inputs(os.path.join(HERE, "data", "inputs"))
out = F.build_features(modis, era)
X = out["X"]
np.savez_compressed(os.path.join(HERE, "output", "features.npz"), X=X,
                    step_start=out["step_start"], step_end=out["step_end"],
                    channels=np.array(list(C.CHANNELS) + ["modis_observed"]))
print("X", X.shape, X.dtype, "| channels:", list(C.CHANNELS) + ["modis_observed"])
for i, n in enumerate(list(C.CHANNELS) + ["modis_observed"]):
    v = X[:, i]; print(f"  {n:15s} mean {v.mean():7.3f} std {v.std():6.3f} min {v.min():6.2f} max {v.max():6.2f}")
print("share of cell-steps with MODIS unobserved:", round(float(1 - X[:, -1].mean()), 4))
clipped = {n: float((abs(X[:, i]) >= C.Z_CLIP - 1e-6).mean()) for i, n in enumerate(C.CHANNELS)}
print("share clipped at +/-%.0f:" % C.Z_CLIP, {k: round(v, 4) for k, v in clipped.items()})

# usable samples per block: label window in block AND 12 steps of history available
lab = np.load(os.path.join(HERE, "output", "labels.npz"), allow_pickle=True)
from datetime import date
origins = [date.fromisoformat(s) for s in lab["origins"]]
ok = np.array([F.input_window(out["step_end"], o) is not None for o in origins])
blocks = lab["block"]
print("origins with full input history:", int(ok.sum()), "of", len(origins))
for b in ("train", "val", "test"):
    print(" ", b, [int(((blocks[:, j] == b) & ok).sum()) for j in range(3)], "samples per horizon (28/42/56 d)")
