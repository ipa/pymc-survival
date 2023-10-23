import sys
import os
import argparse
import joblib
import pandas as pd
from tqdm import tqdm
from pmsurv.models.exponential_model import ExponentialModel
from pmsurv.models.weibull_linear import WeibullModelLinear
from pmsurv.models.weibull_nn import WeibullModelNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('--results-dir', default='results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    experiment_name = args.experiment
 
    results_file = os.path.join(args.results_dir, experiment_name, 'results.csv')
    results = pd.read_csv(results_file)

    results_new = pd.DataFrame(
            columns=['experiment', 'starttime', 'run', 
                'cindex_orig', 'cindex_train', 'cindex_test', 'cindex_diff'])

    for run_idx, run_row in tqdm(results.iterrows()):
        run_path = os.path.join(args.results_dir, experiment_name, str(run_row.starttime), str(run_row.run))
    
        data = joblib.load(os.path.join(run_path, 'data.pkl'))
        X_train, X_test, y_train, y_test = data
    
        if 'exp' in experiment_name or 'wb' in experiment_name:
            selector = joblib.load(os.path.join(run_path, 'selector.pkl'))
            if 'exp_' in experiment_name:
                model = ExponentialModel()
            elif 'nnwb_' in experiment_name:
                model = WeibullModelNN(k_constant=True)
            elif 'wb_' in experiment_name:
                model = WeibullModelLinear(k_constant=True)
        
            model.load(os.path.join(run_path, 'model.yaml'))
        
            cindex_train = model.score(selector.transform(X_train), y_train)
            cindex_test = model.score(selector.transform(X_test), y_test)
        
        else:
            model = joblib.load(os.path.join(run_path, 'model.pkl'))
        
            cindex_train = model.score(X_train, y_train)
            cindex_test = model.score(X_test, y_test)
        
        cindex_diff = cindex_train - cindex_test
    
        results_new = pd.concat([results_new, pd.DataFrame({
            'experiment': [experiment_name],
            'starttime': [run_row.starttime], 
            'run': [run_row.run], 
            'cindex_orig': [run_row.cindex], 
            'cindex_train': [cindex_train], 
            'cindex_test': [cindex_test], 
            'cindex_diff': [cindex_diff]
        })])

        results_new.to_csv(os.path.join(args.results_dir, experiment_name, 'results_overfit.csv'))


if __name__ == "__main__":
    main()


