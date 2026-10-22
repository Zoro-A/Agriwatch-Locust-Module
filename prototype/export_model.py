"""Export the final rolling-refit ConvLSTM from a run checkpoint into a standalone .pt file.

python3 prototype/export_model.py output/ckpt_any_convlstm_sight_month_s1.pkl [seed]
The checkpoint holds the network from the LAST refit (all labels known at its cut-off).
"""
import os, pickle, re, sys
import torch

src = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
st = pickle.load(open(src, "rb"))
assert st["next"] == 8, f"run not finished: next refit index is {st['next']}"
m = re.search(r"cutoff (\d{4}-\d{2}-\d{2}).*best val PR-AUC ([0-9.]+)", st["log"][-1])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "convlstm_any_sight_month.pt")
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(dict(kind="convlstm", variant="sight_month", seed=seed, state=st["state"], cutoff=m.group(1),
                best_val=float(m.group(2)), name="ConvLSTM (2 layers 32/64, 3x3, three horizon heads, 265,539+ parameters)"), out)
print("saved", out, "cutoff", m.group(1), "val PR-AUC", m.group(2))
