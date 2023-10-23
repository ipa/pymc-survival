import argparse
import logging
import time
import warnings
import sys
import contextlib
import os
import utils
import numpy as np
from pqdm.processes import pqdm
import pmsurv_exponential
import pmsurv_weibull
import deepsurv
import pmsurv_common


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

save_fun = {
    'pmsurv_exponential': pmsurv_common.save,
    'pmsurv_weibull_linear': pmsurv_common.save,
    'deepsurv': deepsurv.save
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
    parser.add_argument('--start-run', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=1, help='nr of parallel processes')
    parser.add_argument('--n-partitions', type=int, default=25, help='number of iterations in search')
    return parser.parse_args()


def full_train(partition, model, config, kwargs):
    logger.info("full train")
    models_fulltrain = {}
    score_fulltrain = []
    n_partitions = kwargs['n_partitions']
    for i in range(1, n_partitions):
        logger.info(f'partition {i}')
        X_train, y_train = preprocess_data_fun[model](partition[100+i], config)
        X_test, y_test = preprocess_data_fun[model](partition[i+1], config)
        logger.info(f'shape {X_train.shape}, {X_test.shape}')
        if config['preprocessing']['standardize']:
            logger.debug("Standardize data")
            X_train, X_test, _ = utils.standardize(X_train, X_test, config)
            
        models_fulltrain[i] = retrain_fun[model](X_train, y_train, config, train_kwargs)
        score_ = models_fulltrain[i].score(X_test, y_test)
        score_fulltrain.append(score_)
        y_predicted = models_fulltrain[i].predict(X_test)
        save_data = (X_train, X_test, y_train, y_test, y_predicted)
        save_fun[model](os.path.join(kwargs['experiment_dir'], 'full', str(i)), {'model': models_fulltrain[i]}, score_, {}, save_data)

    return score_fulltrain


def re_train(partition, model, config, kwargs):
    logger.info("re train")
    models = {}
    score_retrain = []
    n_partitions = kwargs['n_partitions']
    for i in range(1, n_partitions):
        logger.info(f'partition {i}')
        X_train, y_train = preprocess_data_fun[model](partition[i], config)
        X_test, y_test = preprocess_data_fun[model](partition[i+1], config)
        logger.debug(f'shape {X_train.shape}, {X_test.shape}')
        if config['preprocessing']['standardize']:
            logger.debug("Standardize data")
            X_train, X_test, _ = utils.standardize(X_train, X_test, config)
        
        if i == 1:
            models[i] = retrain_fun[model](X_train, y_train, config, train_kwargs, prior_model=None)
        else:
            models[i] = retrain_fun[model](X_train, y_train, config, train_kwargs, prior_model=models[i-1])
        
        score_ = models[i].score(X_test, y_test)
        score_retrain.append(score_)
        y_predicted = models[i].predict(X_test)
        save_data = (X_train, X_test, y_train, y_test, y_predicted)
        save_fun[model](os.path.join(kwargs['experiment_dir'], 'retrain',  str(i)), {'model': models[i]}, score_, {}, save_data)

    return score_retrain


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

    score_fulltrain = full_train(partition, model, config, train_kwargs)
    score_retrain = re_train(partition, model, config, train_kwargs)
    return score_fulltrain, score_retrain


if __name__ == '__main__':
    working = os.environ.get("WORKING_DIRECTORY",  os.path.dirname(os.path.abspath(__file__)))
    os.chdir(working)

    logging.basicConfig(level=logging.WARNING)
    logger_pymc = logging.getLogger("pymc")
    logger_pymc.setLevel(logging.INFO)
    logger_pymc.propagate = False

    args = parse_args()

    logger.info("Running {0} experiment".format(args.experiment))

    start_time = time.strftime("%y%m%d%H%M", time.localtime())

    # Load Dataset
    logger.info("Loading datasets: " + args.dataset)
    dataset, config = utils.load_data(args.dataset)

    logger.info(f"Running for {args.start_run} to {args.runs} runs..")

    experiment_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    train_kwargs = {'jobs': args.jobs, 'n_partitions': args.n_partitions}

    warnings.filterwarnings(action='ignore')
    
    run_args = []
    for run in range(args.start_run, args.runs):
        run_args.append({'run': run, 'model': args.model, 'dataset': dataset, 
                         'dataset_name': args.dataset, 'config': config, 'train_kwargs': train_kwargs, 
                         'experiment_dir': experiment_dir, 'start_time': start_time})

    def fun(a):
        np.random.seed(100+int(a['run']))
        try:
            a['train_kwargs']['experiment_dir'] = os.path.join(a['experiment_dir'], str(a['run']))
            c_index_full, c_index_retrain = run_experiment(a['model'], a['dataset'], a['config'], a['train_kwargs'])
            utils.save_results_retrain(a['experiment_dir'], a['model'], a['dataset_name'], c_index_full, 'full')
            utils.save_results_retrain(a['experiment_dir'], a['model'], a['dataset_name'], c_index_retrain, 'retrain')
            utils.save_results_retrain(a['train_kwargs']['experiment_dir'], a['model'], a['dataset_name'], c_index_full, 'full')
            utils.save_results_retrain(a['train_kwargs']['experiment_dir'], a['model'], a['dataset_name'], c_index_retrain, 'retrain')
            return True #[c_index_full[i] - c_index_retrain[i] for i in range(len(c_index_full))]
        except:
            logger.error("Something failed")
            logger.error('Unexpected error: ', sys.exc_info())
            return False
            
    
    result = pqdm(run_args, fun, n_jobs=args.jobs) #, exception_behaviour='immediate'
    logger.info(f'{np.sum(result)} / {len(result)} succeeded')

    logger.info("Finished")

