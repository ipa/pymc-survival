import warnings

import joblib
import matplotlib
import argparse
import os

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sksurv.datasets import get_x_y
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import as_concordance_index_ipcw_scorer

from lifelines.utils import concordance_index

import logging
import time
import utils
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import yaml
from tqdm import tqdm
import numpy as np
from scipy import stats as sp
matplotlib.use('Agg')
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

localtime = time.localtime()
TIMESTRING = time.strftime("%m%d%Y%M", localtime)

DURATION_COL = 'time'
EVENT_COL = 'censor'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='name of the experiment that is being run')
    parser.add_argument('dataset', help='.h5 File containing the train/valid/test datasets')
    parser.add_argument('--results_dir', default='results',
                        help='Directory to save resulting models and visualizations')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    return parser.parse_args()


def run_rsf(X, y, config, kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=config['split']['test_size'],
                                                        shuffle=config['split']['shuffle'])

    if config['preprocessing']['standardize']:
        logger.info("Standardize data")
        X_train, X_test = utils.standardize(X_train, X_test, config)

    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', RandomSurvivalForest())
        ]
    )

    parameters = {
        'model__n_estimators': Integer(5, 25),
        'model__max_depth': Categorical([3, 5, 7, 9]),
        'selector__k': Integer(1, X_train.shape[1]),
    }

    jobs = 1 if 'jobs' not in kwargs else kwargs['jobs']
    opt = BayesSearchCV(pipeline, parameters, n_jobs=jobs, n_points=2, n_iter=50, cv=5)
    opt.fit(X_train, y_train)

    metrics = opt.best_estimator_.score(X_test, y_test)
    logger.info("Test metrics: " + str(metrics))

    return metrics, opt.best_estimator_, opt.best_params_, (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    logger.info("Running {0} experiment".format(args.experiment))
    # logger.info("Arguments:", str(args))

    # Load Dataset
    logger.info("Loading datasets: " + args.dataset)
    dataset, config = utils.load_data(args.dataset)

    X = dataset[config['features']]
    _, y = get_x_y(dataset,
                   attr_labels=[config['outcome']['event'], config['outcome']['time']],
                   pos_label=config['outcome']['pos_label'])

    logger.info(f"Running for {args.runs} runs..")
    pbar = tqdm(range(args.runs))

    experiment_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    c_indexes = utils.RollingMean()
    warnings.filterwarnings(action='ignore')
    for run in pbar:
        c_index, model, params, data = run_rsf(X, y, config, {'jobs': args.jobs})
        c_indexes.add(c_index)
        utils.save_results(args.results_dir, 'rsf', args.dataset, c_index, params)
        pbar.set_description(f"C-Index {str(c_indexes)}")

        if c_index >= c_indexes.get_max():
            data_dump = {'model': model, 'params': params, 'data': data}
            joblib.dump(data_dump, os.path.join(experiment_dir, f"rsf_best.pkl"))

