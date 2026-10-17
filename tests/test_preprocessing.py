import numpy as np
from numpy.testing import assert_allclose
from pmsurv.preprocessing import LogStandardScaler, SelectedColumnsRobustScaler


def test_log_standard_scaler_roundtrip():
    # small deterministic dataset
    x = np.array([[1., 10., 100.],
                  [2., 20., 200.],
                  [3., 30., 300.],
                  [4., 40., 400.]])
    sel = [1]  # scale only column 1
    scaler = LogStandardScaler(selected_columns=sel)
    scaler.fit(x)
    xt = scaler.transform(x)
    xr = scaler.reverse(xt)
    # Expect selected columns approximately equal after reverse
    assert_allclose(x[:, sel], xr[:, sel], rtol=1e-6, atol=1e-6)
    # Ensure non-selected columns unchanged
    assert_allclose(x[:, [0,2]], xr[:, [0,2]], rtol=1e-12, atol=1e-12)


def test_selected_columns_robust_scaler_shape_and_stability():
    x = np.array([[0., 1., 2.],
                  [0., 2., 4.],
                  [0., 3., 6.]])
    sel = [1, 2]
    scaler = SelectedColumnsRobustScaler(selected_columns=sel)
    scaler.fit(x)
    xt = scaler.transform(x)
    # shape unchanged
    assert xt.shape == x.shape
    # selected columns should be numeric and finite
    assert np.isfinite(xt[:, sel]).all()
