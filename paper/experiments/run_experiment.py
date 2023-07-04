import argparse
import logging
import time
import warnings
import sys
import contextlib
import os
import joblib
from tqdm import tqdm
from pqdm.processes import pqdm
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper
import utils
import pmsurv_exponential
import pmsurv_weibull
import pmsurv_weibull_nn
import pmsurv_gp
import pmsurv_common
import rsf
import coxph
import deepsurv
import numpy as np
import pandas as pd

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


preprocess_data_fun = {
    'pmsurv_exponential': pmsurv_exponential.preprocess_data,
    'pmsurv_weibull_linear': pmsurv_weibull.preprocess_data,
    'pmsurv_weibull_nn': pmsurv_weibull_nn.preprocess_data,
    'pmsurv_gp': pmsurv_common.preprocess_data,
    'rsf': rsf.preprocess_data,
    'cox': coxph.preprocess_data,
    'deepsurv': deepsurv.preprocess_data
}

train_fun = {
    'pmsurv_exponential': pmsurv_exponential.train_model,
    'pmsurv_weibull_linear': pmsurv_weibull.train_model,
    'pmsurv_weibull_nn': pmsurv_weibull_nn.train_model,
    'pmsurv_gp': pmsurv_gp.train_model,
    'rsf': rsf.train_model,
    'cox': coxph.train_model,
    'deepsurv': deepsurv.train_model
}

save_fun = {
    'pmsurv_exponential': pmsurv_exponential.save,
    'pmsurv_weibull_linear': pmsurv_exponential.save,
    'pmsurv_weibull_nn': pmsurv_exponential.save,
    'pmsurv_gp': pmsurv_common.save,
    'rsf': rsf.save,
    'cox': coxph.save,
    'deepsurv': deepsurv.save
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of the model')
    parser.add_argument('experiment', help='name of the experiment that is being run')
    parser.add_argument('dataset', help='folder containeing dataset (data.csv)')
    parser.add_argument('--results-dir', default='results', help='directory to save results')
    parser.add_argument('--runs', type=int, default=1, help='repetitions of experiments')
    parser.add_argument('--jobs', type=int, default=1, help='nr of parallel processes')
    parser.add_argument('--n-iter', type=int, default=25, help='number of iterations in search')
    return parser.parse_args()

def cb_log(res):
    logger.info(f'C-Index: {-res.fun}')

def run_experiment(model, dataset, config, train_kwargs):
    dataset = dataset.dropna(subset=config['features'])
    X, y = preprocess_data_fun[model](dataset, config)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=config['split']['test_size'],
                                                        shuffle=config['split']['shuffle'])

    if config['preprocessing']['standardize']:
        logger.info("Standardize data")
        X_train, X_test, _ = utils.standardize(X_train, X_test, config)

    pipeline, parameters, fit_params = train_fun[model](X_train, y_train, config, train_kwargs)
    n_cv = 5
    n_points = 1 
    opt = BayesSearchCV(pipeline, parameters,
                        fit_params=fit_params,
                        n_jobs=1,
                        n_points=n_points if n_points > 1 else 1,
                        n_iter=train_kwargs['n_iter'],
                        cv=n_cv,
                        error_score='raise')
    stopper = DeltaYStopper(0.02, n_best=6)
    opt.fit(X_train, y_train, callback=[stopper, cb_log])
    metrics = opt.best_estimator_.score(X_test, y_test)
    logger.info("Test metrics: " + str(metrics))

    return metrics, opt.best_estimator_, opt.best_params_, (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    working = os.environ.get("WORKING_DIRECTORY",  os.path.dirname(os.path.abspath(__file__)))
    os.chdir(working)

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO)
    logger_pymc = logging.getLogger("pymc")
    logger_pymc.setLevel(logging.ERROR)
    logger_pymc.propagate = False

    args = parse_args()

    logger.info("Running {0} experiment".format(args.experiment))

    # Load Dataset
    logger.info("Loading datasets: " + args.dataset)
    dataset, config = utils.load_data(args.dataset)

    logger.info(f"Running for {args.runs} runs...")
    pbar = tqdm(range(args.runs))

    start_time = time.strftime("%y%m%d%H%M", time.localtime())

    experiment_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    train_kwargs = {'jobs': args.jobs, 'n_iter': args.n_iter}
    logger.info(f"train arguments {train_kwargs}")
    c_indexes = utils.RollingMean()
    warnings.filterwarnings(action='ignore')

    run_args = []
    for run in range(args.runs):
        run_args.append({'run': run, 'model': args.model, 'dataset': dataset, 'dataset_name': args.dataset, 
                         'config': config, 'train_kwargs': train_kwargs, 'experiment_name': args.experiment,
                         'experiment_dir': experiment_dir, 'start_time': start_time})

    def fun(a):
        try:
            np.random.seed(int(a['run']))
            c_index, model, params, data = run_experiment(a['model'], a['dataset'], a['config'], a['train_kwargs'])
            
            X_train, X_test, y_train, y_test = data
            cindex_train = model.score(X_train, y_train)
            cindex_test = model.score(X_test, y_test)

            utils.save_results(a['experiment_dir'], a['model'], a['dataset_name'], c_index, cindex_train, cindex_test, params, a['start_time'], a['run'])
            save_fun[a['model']](os.path.join(a['experiment_dir'], str(a['start_time']), str(a['run'])), model, c_index, params, data)

            return True
        except:
            logger.error('Unexpected error: ', sys.exc_info()[0])
            return False
    
    result = pqdm(run_args, fun, n_jobs=args.jobs) 

    logger.info(f'{np.sum(result)} / {len(result)} succeeded')
    logger.info("Finished")
