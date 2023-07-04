import os
import logging
import joblib
import yaml
import lifelines
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import pandas as pd
try:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH
except:
    print("could not load pycox")
import numpy as np
from skopt.space import Real, Categorical, Integer
import utils

logger = logging.getLogger(__name__)

class PyCoxWrapper(BaseEstimator):

    def __init__(self, hidden_units=None, dropout=0.1, lr=1e-3):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.net = None 
        self.cox = None

    def fit(self, X, y, batch_size=512, epochs=200):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        y = (y[:, 0], y[:, 1])

        # self.hidden_units = [X.shape[1], X.shape[1]]
        #if batch_size > (X.shape[0] / 2):
        #    batch_size = 128
        if self.cox is None:
            logger.info('learn from scratch')
            self.net = tt.practical.MLPVanilla(in_features=X.shape[1],
                                      num_nodes=[self.hidden_units],
                                      out_features=1,
                                      batch_norm=True,
                                      dropout=self.dropout,
                                      output_bias=False)
            self.cox = CoxPH(self.net, tt.optim.Adam(1e-2), )
            self.cox.optimizer.set_lr(self.lr)

        log = self.cox.fit(X, y, batch_size, epochs,
                           verbose=False)
        _ = self.cox.compute_baseline_hazards()
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        else:
            X = X.astype(np.float32)
        surv = self.cox.predict_surv_df(X).T.values
        return surv

    def score(self, X, y):
        y_pred = self.predict(X.astype(np.float32))
        y_pred = np.nanmedian(y_pred, axis=1)
        #print(y_pred)
        c_index = lifelines.utils.concordance_index(y[:, 0], y_pred, y[:, 1])
        return c_index


def preprocess_data(dataset, config):
    X = dataset[config['features']]
    y = dataset[[config['outcome']['time'], config['outcome']['event']]].values
    # y = y.astype(np.float32)
    # y[:, 1] = 1 - y[:, 1]  # inverse to censored
    return X, y


def train_model(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', PyCoxWrapper())
        ]
    )

    parameters = {
        'model__hidden_units': Integer(2, X_train.shape[1]),
        'model__lr': Real(1e-5, 1e-1),
        'model__dropout': Real(0.1, 0.5),
        'selector__k': Integer(1, X_train.shape[1]),
    }

    fit_params = {}

    return pipeline, parameters, fit_params


def retrain_model(X_train, y_train, config, train_kwargs, prior_model=None):
    import copy

    if prior_model is None:
        model = PyCoxWrapper(hidden_units=X_train.shape[1])
    else:
        model = copy.copy(prior_model)
    
    model.fit(X_train.values, y_train)
    return model


def save(save_dir, model, c_index, params, data):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, "model.pkl"))
    joblib.dump(data, os.path.join(save_dir, "data.pkl"))
    metadata = {
        'c_index': float(c_index),
        'params': dict(params),
    }
    with open(os.path.join(save_dir, "params.yaml"), 'w') as f:
        yaml.dump(metadata, f)



