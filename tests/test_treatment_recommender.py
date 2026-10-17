import numpy as np
import pandas as pd
import tempfile
from pmsurv.treatment_recommender import TreatmentRecommender


class MockSurvivalModel:
    def __init__(self, n_timepoints=5):
        self.n_timepoints = n_timepoints

    def predict(self, x):
        # return surv_prob shaped (n_samples, n_timepoints)
        n = x.shape[0]
        # survival decreasing over time
        surv = np.tile(np.linspace(0.9, 0.1, self.n_timepoints), (n, 1))
        return surv, None, None


def test_find_best_threshold_and_save_load_roundtrip():
    rng = np.random.default_rng(0)
    n = 20
    x = rng.normal(size=(n, 3))
    # y: time, event where event==0 indicates event occurred in this codebase
    times = rng.uniform(1, 10, size=n)
    events = rng.integers(0, 2, size=n)
    y = np.vstack((times, events)).T

    model = MockSurvivalModel(n_timepoints=5)
    recommender = TreatmentRecommender(model)
    thresholds, thresholds_dict = recommender.find_best_threshold(x, y)
    # ensure keys present
    assert 'best' in thresholds_dict and 'best_tpr' in thresholds_dict and 'best_fpr' in thresholds_dict and 'low_risk' in thresholds_dict

    # test save/load
    recommender.thresholds = {k: float(v) for k, v in thresholds_dict.items()}
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_name = tmp.name
    tmp.close()
    recommender.save(tmp_name)

    new_rec = TreatmentRecommender(model)
    new_rec.load(tmp_name)
    # loaded thresholds should equal saved ones (json stores as strings/numbers)
    for k in recommender.thresholds:
        assert k in new_rec.thresholds


def test_predict_risk_classification():
    rng = np.random.default_rng(1)
    n = 6
    x = np.ones((n, 3))
    times = np.arange(1, n+1).astype(float)
    events = np.zeros(n, dtype=int)
    y = np.vstack((times, events)).T

    model = MockSurvivalModel(n_timepoints=5)
    recommender = TreatmentRecommender(model)
    # set threshold such that half are low, half high based on median_risk
    recommender.thresholds = {'low_risk': 0.5}
    df = recommender.predict_risk(x, y=y, threshold_idx='low_risk')
    assert 'risk' in df.columns
    assert df.shape[0] == n
