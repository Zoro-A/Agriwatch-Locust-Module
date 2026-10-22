import numpy as np
import pytest

import config as C
import metrics as Mx
from dataset import Samples
import baselines as B


def _grid(n=4):
    P = np.zeros((n, C.N_ROWS, C.N_COLS)); Y = np.zeros_like(P, dtype=np.uint8); M = np.ones_like(Y)
    return P, Y, M


def test_perfect_ranking_gives_pr_auc_one_and_best_f1_one():
    P, Y, M = _grid()
    Y[:, 3, 3] = 1; P[:, 3, 3] = 0.9; P[:, 5, 5] = 0.1
    assert Mx.pr_auc(P, Y, M) == pytest.approx(1.0)
    t, f1 = Mx.best_f1_threshold(P, Y, M)
    assert f1 == pytest.approx(1.0) and 0.1 < t <= 0.9


def test_unknown_cells_are_ignored():
    P, Y, M = _grid(2)
    Y[:, 3, 3] = 1; P[:, 3, 3] = 0.9
    P[:, 7, 7] = 1.0; M[:, 7, 7] = 0          # high score on an unknown cell must not count
    assert Mx.pr_auc(P, Y, M) == pytest.approx(1.0)
    assert Mx.recall_at_alert_area(P, Y, M, 1) == pytest.approx(1.0)


def test_recall_at_alert_area_counts_top_q_percent_of_known_cells():
    P, Y, M = _grid(1)
    n = C.N_ROWS * C.N_COLS
    rng = np.random.default_rng(0); P[0] = rng.random((C.N_ROWS, C.N_COLS))
    top = np.unravel_index(np.argsort(-P[0].ravel())[: n // 10], P[0].shape)
    Y[0][top[0][:5], top[1][:5]] = 1                   # 5 positives, all inside the top 10%
    assert Mx.recall_at_alert_area(P, Y, M, 10) == pytest.approx(1.0)
    assert Mx.recall_at_alert_area(-P, Y, M, 10) == pytest.approx(0.0)


def test_tolerant_metrics_credit_adjacent_cells():
    P, Y, M = _grid(1)
    Y[0, 10, 10] = 1; P[0, 10, 11] = 1.0               # flagged cell is one cell east of the positive
    assert Mx.recall_at_alert_area(P, Y, M, 0.2) == 0.0
    assert Mx.tolerant_recall_at_alert_area(P, Y, M, 0.2) == 1.0
    assert Mx.tolerant_pr_auc(P, Y, M) > Mx.pr_auc(P, Y, M)


def test_brier_value():
    P, Y, M = _grid(1)
    Y[0, 0, 0] = 1; P[:] = 0.5
    assert Mx.brier(P, Y, M) == pytest.approx(0.25)


def test_bootstrap_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    P = rng.random((12, C.N_ROWS, C.N_COLS)); Y = (P + rng.normal(0, .3, P.shape) > 0.8).astype(np.uint8)
    M = np.ones_like(Y)
    a = Mx.block_bootstrap(Mx.pr_auc, P, Y, M, n_boot=200, seed=3)
    b = Mx.block_bootstrap(Mx.pr_auc, P, Y, M, n_boot=200, seed=3)
    assert a == b and a[0] <= Mx.pr_auc(P, Y, M) <= a[1]


@pytest.fixture(scope="module")
def S():
    import os
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "features.npz")):
        pytest.skip("run build_labels.py and build_features.py first")
    return Samples("any")


def test_each_sample_has_one_block_and_valid_labels(S):
    assert set(S.block) == set(C.BLOCKS)
    assert (S.Y <= S.M).all()
    assert S.window(0).shape == (C.SEQ_LEN, len(C.CHANNELS) + 1, C.N_ROWS, C.N_COLS)
    keys = set(zip(S.origin_idx.tolist(), S.hor_idx.tolist()))
    assert len(keys) == len(S.origin_idx)             # no (origin, horizon) pair appears twice


def test_inputs_never_extend_past_origin_and_train_origins_precede_val(S):
    for k in range(len(S.origin_idx)):
        assert S.step_end[S.win_stop[k] - 1] == S.origins[S.origin_idx[k]].isoformat()
    tr = [S.origins[S.origin_idx[k]] for k in np.where(S.block == "train")[0]]
    assert max(tr) <= C.BLOCKS["train"][1]


def test_b0_prior_is_a_probability_map_with_sensible_mean(S):
    tr, va = S.select("train", 0), S.select("val", 0)
    m = B.ClimatologicalPrior().fit(S, tr)
    P = m.predict(S, va)
    assert P.shape == (len(va), C.N_ROWS, C.N_COLS) and P.min() >= 0 and P.max() <= 1
    train_prev = S.Y[tr].sum() / S.M[tr].sum()
    assert 0.2 * train_prev < P.mean() < 5 * train_prev
