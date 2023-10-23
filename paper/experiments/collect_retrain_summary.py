import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import arviz as az

EXCLUDES = ['lambda_det', 'k_det']

def overlap_index(trace1, trace2):
    cmin = np.min([trace1, trace2]) - 0.1
    cmax = np.max([trace1, trace2]) + 0.1
    grid1, pdf1 = az.kde(trace1, custom_lims=(cmin, cmax))
    grid2, pdf2 = az.kde(trace2, custom_lims=(cmin, cmax))
    
    overlap = np.min(np.stack([pdf1, pdf2]), axis=0) 
    
    auc_1 = -np.trapz(grid1, pdf1)
    auc_2 = -np.trapz(grid2, pdf2)
    #iscale_factor = 1 / np.mean([auc_1, auc_2])
    auc_overlap = -np.trapz(grid1, overlap) 
    
    return auc_overlap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='base directory')
    parser.add_argument('run_from', type=int)
    parser.add_argument('run_to', type=int)
    parser.add_argument('partition_from', type=int)
    parser.add_argument('partition_to', type=int)
    return parser.parse_args()
    

def collect_ovi(experiment_dir, run_from, run_to, partition_range):
    results = pd.DataFrame({ 'model': [],  'run': [],  'partition': [],  'var': [], 'ovi': [] })

    pbar = tqdm(range(run_from, run_to))
    for run in pbar:
        run_dir = os.path.join(experiment_dir, str(run))
        for partition in partition_range:
            pbar.set_description(f"{run} / {partition}")
            try:
                file_retrain = os.path.join(run_dir, 'retrain', str(partition), 'model.yaml.netcdf')
                file_full = os.path.join(run_dir, 'full', str(partition), 'model.yaml.netcdf')

                trace_retrain = az.from_netcdf(file_retrain)
                trace_full = az.from_netcdf(file_full)
            except:
                print(f"{file_retrain} failed...")
                print(f"{file_full} failed...")
                continue
        
            for i, d_var in enumerate(trace_retrain.posterior.data_vars):
                if d_var in EXCLUDES:
                    continue
        
                trace_var_retrain = trace_retrain.posterior[d_var].values.flatten()
                trace_var_full = trace_full.posterior[d_var].values.flatten()
                ovi = overlap_index(trace_var_retrain, trace_var_full)
            
                results = pd.concat([results, pd.DataFrame({
                    'model': ['pm_exp'],  
                    'run': [run],  
                    'partition': [partition],  
                    'var': [d_var],  
                    'posterior_retrain': [trace_var_retrain],  
                    'posterior_full': [trace_var_full], 
                    'ovi': [ovi] 
                    })
                ])

    return results



def collect_posterior(results, run):
    results_posterior = pd.DataFrame({ 'model': [], 'run': [], 'partition': [], 'var': [],  'posterior_type': [],  'posterior': []})

    results_run = results[results['run'] == run]

    for i in tqdm(range(0, results_run.shape[0])):
        for post_type in ['posterior_retrain', 'posterior_full']:
            n_posterior = len(results_run.iloc[i][post_type])
            results_posterior = pd.concat([results_posterior, pd.DataFrame({
                'model': np.repeat(results_run.iloc[i]['model'], n_posterior),
                'run': np.repeat(results_run.iloc[i]['run'], n_posterior),
                'partition': np.repeat(results_run.iloc[i]['partition'], n_posterior),
                'var': np.repeat(results_run.iloc[i]['var'], n_posterior),
                'posterior_type': np.repeat(post_type, n_posterior),
                'posterior_sampled': results_run.iloc[i][post_type]})
            ])

    return results_posterior


def main():
    args = parse_args()
    print(args)
    
    results_ovi = collect_ovi(args.dir, args.run_from, args.run_to, range(args.partition_from, args.partition_to))

    results_ovi.to_csv(os.path.join(args.dir, 'ovi.csv'))

    results_posterior_5 = collect_posterior(results_ovi, 5)
    results_posterior_5.to_csv(os.path.join(args.dir, 'posterior_5.csv'))

if __name__ == "__main__":
    main()
