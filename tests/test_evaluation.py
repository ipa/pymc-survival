import numpy as np
from pmsurv.evaluation import bootstrap_metric, evaluate_model


def test_bootstrap_metric_basic():
    # simple metric function that returns mean of first feature
    def metric_fxn(x, y):
        return float(np.mean(x[:, 0]))

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = rng.normal(size=(20, 1))

    res = bootstrap_metric(metric_fxn, x, y, N=20)
    assert 'mean' in res and 'confidence_interval' in res
    ci = res['confidence_interval']
    assert isinstance(ci, tuple) and len(ci) == 2


class DummyModel:
    def score(self, x, y):
        return 0.75


def test_evaluate_model_returns_c_index():
    model = DummyModel()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 2))
    y = np.vstack((np.arange(10).astype(float), (rng.integers(0,2,size=10))).T)

    metrics = evaluate_model(model, x, y, bootstrap=False)
    assert 'c_index' in metrics
    assert metrics['c_index'] == 0.75
