from datetime import date
import numpy as np
import torch
import pytest

import config as C
import nn_models as N


def test_param_counts_match_report():
    assert N.n_params(N.ConvLSTMNet()) == 43_904 + 221_440 + 195 == 265_539


@pytest.mark.parametrize("cls", [N.ConvLSTMNet, N.CellLSTMNet])
def test_output_shape(cls):
    y = cls()(torch.zeros(2, C.SEQ_LEN, 6, C.N_ROWS, C.N_COLS))
    assert y.shape == (2, 3, C.N_ROWS, C.N_COLS)


def test_masked_loss_ignores_unknown_cells():
    z = torch.randn(1, 3, 4, 4)
    Y = (torch.rand(1, 3, 4, 4) > .5).float()
    M = torch.zeros_like(Y); M[0, 0, 0, 0] = 1
    a = N.masked_bce(z, Y, M, 3.0)
    z2 = z.clone(); z2[0, 1:] += 100          # changing masked-out cells must not change the loss
    assert torch.isclose(a, N.masked_bce(z2, Y, M, 3.0))


def test_convlstm_cell_is_translation_equivariant_for_constant_input():
    torch.manual_seed(0)
    m = N.ConvLSTMNet().eval()
    y = m(torch.ones(1, 3, 6, 10, 10))
    assert torch.allclose(y[0, :, 5, 5], y[0, :, 4, 5], atol=1e-4)    # interior cells identical


def test_fit_uses_no_future_labels():
    from dataset import Samples
    import rolling as R
    S = Samples("any", include_all=True)
    D = N.OriginData(S)
    cut = date(2019, 6, 1)
    for j in range(3):
        for k in R.train_indices(S, j, cut):
            assert S.tw_end[k] <= cut


def test_input_variants_shapes_and_no_future_sightings():
    from dataset import Samples
    S = Samples("any", include_all=True)
    k = len(S.origin_idx) // 2
    for v, c in N.VARIANTS.items():
        x = N.build_input(S, k, v)
        assert x.shape == (C.SEQ_LEN, c, C.N_ROWS, C.N_COLS) and np.isfinite(x).all()
    # the last input step must end on or before the forecast origin
    assert str(S.step_end[S.win_stop[k] - 1]) <= str(S.origin_date[k])
