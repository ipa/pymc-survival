import argparse
import logging
import time
import warnings
import sys
import contextlib
import os
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
import utils
import pmsurv_exponential
import pmsurv_weibull
import rsf

localtime = time.localtime()
TIMESTRING = time.strftime("%m%d%Y%M", localtime)


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
    'rsf': rsf.preprocess_data
}

train_fun = {
    'pmsurv_exponential': pmsurv_exponential.train_model,
    'pmsurv_weibull_linear': pmsurv_weibull.train_model,
    'pmsurv_weibull_linear_k': pmsurv_weibull.train_model,
    'pmsurv_weibull_nn': pmsurv_weibull.train_model,
    'pmsurv_weibull_nn_k': pmsurv_weibull.train_model,
    'rsf': rsf.train_model,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of the model')
    parser.add_argument('experiment', help='name of the experiment that is being run')
    parser.add_argument('dataset', help='folder containeing dataset (data.csv)')
    parser.add_argument('--results_dir', default='results', help='directory to save results')
    parser.add_argument('--runs', type=int, default=1, help='repetitions of experiments')
    parser.add_argument('--jobs', type=int, default=1, help='nr of parallel processes')
    parser.add_argument('--n-iter', type=int, default=25, help='number of iterations in search')
    return parser.parse_args()


def run_experiment(model, dataset, config, train_kwargs):
    X, y = preprocess_data_fun[model](dataset, config)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=config['split']['test_size'],
                                                        shuffle=config['split']['shuffle'])

    if config['preprocessing']['standardize']:
        logger.info("Standardize data")
        X_train, X_test = utils.standardize(X_train, X_test, config)

    pipeline, parameters, fit_params = train_fun[model](X_train, y_train, config, train_kwargs)

    opt = BayesSearchCV(pipeline, parameters,
                        fit_params=fit_params,
                        n_jobs=train_kwargs['jobs'],
                        n_points=1,
                        n_iter=train_kwargs['n_iter'],
                        cv=5)
    opt.fit(X_train, y_train)

    metrics = opt.best_estimator_.score(X_test, y_test)
    logger.info("Test metrics: " + str(metrics))

    return metrics, opt.best_estimator_, opt.best_params_, (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.WARNING)
    logger_pymc = logging.getLogger("pymc")
    logger_pymc.setLevel(logging.ERROR)
    logger_pymc.propagate = False

    args = parse_args()

    logger.info("Running {0} experiment".format(args.experiment))

    # Load Dataset
    logger.info("Loading datasets: " + args.dataset)
    dataset, config = utils.load_data(args.dataset)

    logger.info(f"Running for {args.runs} runs..")
    pbar = tqdm(range(args.runs))

    experiment_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    train_kwargs = {'jobs': args.jobs, 'n_iter': args.n_iter}

    c_indexes = utils.RollingMean()
    warnings.filterwarnings(action='ignore')

    for run in pbar:
        with nostdout():
            c_index, model, params, data = run_experiment(args.model, dataset, config, train_kwargs)
        c_indexes.add(c_index)
        utils.save_results(args.results_dir, args.model, args.dataset, c_index, params)
        pbar.set_description(f"C-Index {str(c_indexes)}")

        if c_index >= c_indexes.get_max():
            data_dump = {'params': params, 'data': data}
            joblib.dump(data_dump, os.path.join(experiment_dir, f"{args.model}_best.pkl"))
            try:
                data_dump = {'params': params, 'data': data}
                joblib.dump(data_dump, os.path.join(experiment_dir, f"{args.model}_best.pkl"))
            except:
                logger.error("Failed to save model, need to retrain...")

