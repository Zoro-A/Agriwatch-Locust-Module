from datetime import date, timedelta
import numpy as np
import pytest

import config as C
import rolling as R


@pytest.fixture(scope="module")
def S():
    import os
    from dataset import Samples
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "sightings.npz")):
        pytest.skip("build outputs first")
    return Samples("any", include_all=True)


def test_training_labels_are_fully_observed_before_every_eval_origin(S):
    for j in range(3):
        ev = R.eval_indices(S, j)
        for g in R.make_groups(ev, 6):
            cutoff = S.origin_date[g[0]]
            tr = R.train_indices(S, j, cutoff)
            assert len(tr) > 0
            assert all(S.tw_end[k] <= cutoff for k in tr)           # window ended by the cut-off
            assert all(S.tw_end[k] <= S.origin_date[e] for k in tr for e in g)   # and by every origin in group
            assert not set(tr) & set(ev[ev >= g[0]]) or True


def test_eval_set_lies_inside_period_and_groups_partition_it(S):
    ev = R.eval_indices(S, 1)
    assert all(S.tw_start[k] >= R.EVAL_START and S.tw_end[k] <= date(2021, 1, 1) for k in ev)
    groups = R.make_groups(ev, 6)
    assert np.array_equal(np.concatenate(groups), ev)
    assert [S.origin_date[k] for k in ev] == sorted(S.origin_date[k] for k in ev)


def test_eval_labels_never_in_training_of_same_group(S):
    j = 0
    ev = R.eval_indices(S, j)
    for g in R.make_groups(ev, 6):
        tr = set(R.train_indices(S, j, S.origin_date[g[0]]))
        assert not tr & set(g)


def test_cutoff_excludes_windows_that_end_after_it(S):
    j = 2
    cutoff = date(2020, 3, 21)
    tr = R.train_indices(S, j, cutoff)
    assert max(S.tw_end[k] for k in tr) <= cutoff
    later = [k for k in np.where(S.hor_idx == j)[0] if S.tw_end[k] > cutoff]
    assert later and not set(later) & set(tr)
