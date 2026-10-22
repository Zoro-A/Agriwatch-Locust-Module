from datetime import date

import numpy as np
import pytest

import config as C
import locust_labels as L
import sightings as SG
import baselines as B


def _cubes(events):
    cu = L.EventCubes((C.CALENDAR_END - C.EPOCH).days)
    for names, day, cell in events:
        for nm in names:
            cu.add(nm, day, cell)
    cu.finalize()
    return cu


STARTS = ["2019-08-13", "2019-08-29", "2019-09-14"]
ENDS = ["2019-08-29", "2019-09-14", "2019-09-30"]
A = (5, 5)


def test_record_on_origin_date_is_not_in_the_step_that_ends_at_the_origin():
    cu = _cubes([(("any",), date(2019, 8, 28), A),      # last day of step 0 -> in step 0
                 (("any",), date(2019, 8, 29), A)])     # first day of step 1 = origin of step 0
    S = SG.build_step_sightings(cu, STARTS, ENDS)
    assert S[0, 0, A[0], A[1]] == 1 and S[1, 0, A[0], A[1]] == 1 and S[2, 0, A[0], A[1]] == 0
    cu2 = _cubes([(("any",), date(2019, 8, 29), A)])
    S2 = SG.build_step_sightings(cu2, STARTS, ENDS)
    assert S2[0, 0, A[0], A[1]] == 0                    # origin day itself is excluded from step 0


def test_channels_separate_any_breeding_and_surveyed():
    d = date(2019, 8, 20)
    cu = _cubes([(("any",), d, (1, 1)), (("any", "breeding"), d, (2, 2))])
    cu.add("absence", d, (3, 3)); cu.finalize()
    S = SG.build_step_sightings(cu, STARTS, ENDS)
    assert (S[0, 0, 1, 1], S[0, 1, 1, 1], S[0, 2, 1, 1]) == (1, 0, 1)
    assert (S[0, 0, 2, 2], S[0, 1, 2, 2]) == (1, 1)
    assert (S[0, 0, 3, 3], S[0, 2, 3, 3]) == (0, 1)


class _FakeS:
    def __init__(self, hist):
        self.h = hist
    def sight_window(self, k):
        return self.h


def _hist(positions):
    h = np.zeros((C.SEQ_LEN, 3, C.N_ROWS, C.N_COLS), dtype=np.uint8)
    for t in positions:
        h[t, 0, 10, 10] = 1
    return h


def test_persistence_ranks_recent_records_above_old_ones_and_nonrecords_lowest():
    p = B.Persistence(0)
    recent = p.predict(_FakeS(_hist([C.SEQ_LEN - 1])), [0])[0]
    old = p.predict(_FakeS(_hist([0])), [0])[0]
    assert recent[10, 10] > old[10, 10] > recent[20, 20]
    assert recent.max() <= 1 and recent.min() >= 0
    assert recent[10, 11] > recent[20, 20]               # neighbour gets some credit


@pytest.fixture(scope="module")
def S():
    from dataset import Samples
    import os
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "sightings.npz")):
        pytest.skip("run sightings.py first")
    return Samples("any")


def test_real_sightings_aligned_with_inputs(S):
    k = 100
    assert S.sight_window(k).shape == (C.SEQ_LEN, 3, C.N_ROWS, C.N_COLS)
    assert S.step_end[S.win_stop[k] - 1] == S.origins[S.origin_idx[k]].isoformat()
    # a record counted in the last step lies strictly before the origin
    assert S.sight.shape[0] == S.X.shape[0]
