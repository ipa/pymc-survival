import argparse
import logging
import time
import warnings
import sys
import contextlib
import os
import joblib
from tqdm import tqdm
import utils
import pmsurv_exponential
import pmsurv_weibull
import lifelines
import numpy as np
import multiprocessing
#from pqdm.processes import pqdm
import deepsurv

#TODO: https://docs.pymc.io/en/v3/pymc-examples/examples/pymc3_howto/updating_priors.html

logger = logging.getLogger(__name__)

localtime = time.localtime()
TIMESTRING = time.strftime("%m%d%Y%M", localtime)

preprocess_data_fun = {
    'pmsurv_exponential': pmsurv_exponential.preprocess_data,
    'pmsurv_weibull_linear': pmsurv_weibull.preprocess_data,
    'deepsurv': deepsurv.preprocess_data,
}

retrain_fun = {
    'pmsurv_exponential': pmsurv_exponential.retrain_model,
    'pmsurv_weibull_linear': pmsurv_weibull.retrain_model,
    'deepsurv': deepsurv.retrain_model
}

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='name of the model')
    parser.add_argument('experiment', help='name of the experiment that is being run')
    parser.add_argument('dataset', help='folder containeing dataset (data.csv)')
    parser.add_argument('--results_dir', default='results_retrain', help='directory to save results')
    parser.add_argument('--runs', type=int, default=1, help='repetitions of experiments')
    parser.add_argument('--jobs', type=int, default=1, help='nr of parallel processes')
    parser.add_argument('--n-partitions', type=int, default=25, help='number of iterations in search')
    return parser.parse_args()


def full_train(partition, model, config, n_partitions):
    logger.info("full train")
    models_fulltrain = {}
    score_fulltrain = []
    for i in range(1, n_partitions):
        X_train, y_train = preprocess_data_fun[model](partition[100+i], config)
        X_test, y_test = preprocess_data_fun[model](partition[i+1], config)
        logger.info(f'shape {X_train.shape}, {X_test.shape}')
        if config['preprocessing']['standardize']:
            logger.info("Standardize data")
            X_train, X_test = utils.standardize(X_train, X_test, config)
            
        models_fulltrain[i] = retrain_fun[model](X_train, y_train, config, train_kwargs)
        score_fulltrain.append(models_fulltrain[i].score(X_test, y_test))

    return score_fulltrain


def re_train(partition, model, config, n_partitions):
    logger.info("re train")
    models = {}
    score = []
    for i in range(1, n_partitions):
        X_train, y_train = preprocess_data_fun[model](partition[i], config)
        X_test, y_test = preprocess_data_fun[model](partition[i+1], config)
        logger.debug(f'shape {X_train.shape}, {X_test.shape}')
        if config['preprocessing']['standardize']:
            logger.info("Standardize data")
            X_train, X_test = utils.standardize(X_train, X_test, config)
        
        if i == 1:
            models[i] = retrain_fun[model](X_train, y_train, config, train_kwargs, prior_model=None)
        else:
            models[i] = retrain_fun[model](X_train, y_train, config, train_kwargs, prior_model=models[i-1])
        
        score.append(models[i].score(X_test, y_test))

    return score


def run_experiment(model, dataset, config, train_kwargs):
    dataset = dataset.dropna(subset=config['features'])
    partition = {}
    if config['partition']['type'] == 'var':
        partition_idx = dataset['partition'].values
    else:
        logger.info("do random")
        partition_idx = np.random.randint(train_kwargs['n_partitions'], size=len(dataset)) + 1

    for i in range(1, train_kwargs['n_partitions']+1):
        partition[i] = dataset.iloc[partition_idx==i, :]
        partition[100+i] = dataset.iloc[partition_idx<=i, :]

    score_retrain = re_train(partition, model, config, train_kwargs['n_partitions'])
    score_fulltrain = full_train(partition, model, config, train_kwargs['n_partitions'])
    return score_fulltrain, score_retrain


if __name__ == '__main__':
    working = os.environ.get("WORKING_DIRECTORY",  os.path.dirname(os.path.abspath(__file__)))
    os.chdir(working)

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

    train_kwargs = {'jobs': args.jobs, 'n_partitions': args.n_partitions}

    warnings.filterwarnings(action='ignore')
    
    proc_args = [{'model': args.model, 'dataset': dataset, 'config': config, 'train_kwargs': train_kwargs} for i in range(args.runs)]

    for run in pbar:
        try:
            c_index_full, c_index_retrain = run_experiment(args.model, dataset, config, train_kwargs)
            utils.save_results_retrain(experiment_dir, args.model, args.dataset, c_index_full, 'full')
            utils.save_results_retrain(experiment_dir, args.model, args.dataset, c_index_retrain, 'retrain')
        except:
            logger.warn('one iteration failed')
            raise
    logger.info("Finished")

