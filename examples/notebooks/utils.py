import pandas as pd
import numpy as np
import scipy.stats as st
import yaml
import os
from sklearn.preprocessing import StandardScaler
import scipy as sp

# def load_datasets(dataset_file):
#     datasets = defaultdict(dict)
#
#     with h5py.File(dataset_file, 'r') as fp:
#         for ds in fp:
#             for array in fp[ds]:
#                 datasets[ds][array] = fp[ds][array][:]
#
#     return datasets
#
# def format_dataset_to_df(dataset, duration_col, event_col, trt_idx=None):
#     xdf = pd.DataFrame(dataset['x'])
#     if trt_idx is not None:
#         xdf = xdf.rename(columns={trt_idx: 'treat'})
#
#     dt = pd.DataFrame(dataset['t'], columns=[duration_col])
#     censor = pd.DataFrame(dataset['e'], columns=[event_col])
#     cdf = pd.concat([xdf, dt, censor], axis=1)
#     return cdf
#
#
# def standardize_dataset(dataset, offset, scale, cat_vals=None):
#     norm_ds = copy.deepcopy(dataset)
#     print("standardizing data")
#     print(offset, scale, cat_vals)
#     if cat_vals is None:
#         norm_ds['x'] = (norm_ds['x'] - offset) / scale
#     else:
#         norm_ds_tmp = (norm_ds['x'] - offset) / scale
#         norm_ds_tmp[cat_vals] = norm_ds['x'][cat_vals]
#         norm_ds['x'] = norm_ds_tmp
#     return norm_ds


def standardize(X_train, X_test, config):
    scaler = StandardScaler()
    continuous_features = config['preprocessing']['continuous_features']
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    return X_train, X_test, scaler


def load_data(dataset_folder):
    dataset = pd.read_csv(os.path.join(dataset_folder, 'data.csv'))
    with open(os.path.join(dataset_folder, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    return dataset, config

def bootstrap_metric(metric_fxn, dataset, N=100):
    def sample_dataset(dataset, sample_idx):
        sampled_dataset = {}
        for (key, value) in dataset.items():
            sampled_dataset[key] = value[sample_idx]
        return sampled_dataset

    metrics = []
    size = len(dataset['x'])

    for _ in range(N):
        resample_idx = np.random.choice(size, size=size, replace=True)

        metric = metric_fxn(**sample_dataset(dataset, resample_idx))
        metrics.append(metric)

    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }


def save_results(results_dir, model, experiment, cindex, cindex_train, cindex_test, params, start_time, run):
    os.makedirs(results_dir, exist_ok=True)
    df_results = pd.DataFrame({
        'experiment': [experiment],
        'model': [model],
        'cindex': [cindex],
        'cindex_train': [cindex_train],
        'cindex_test': [cindex_test],
        'cindex_diff': [cindex_train - cindex_test],
        'starttime': [start_time],
        'run': [run],
        'hyperparams': [params]
    })
    result_file = os.path.join(results_dir, "results.csv")
    df_results.to_csv(result_file, mode='a', header=not os.path.exists(result_file))

def save_results_retrain(results_dir, model, experiment, cindex, train_type):
    os.makedirs(results_dir, exist_ok=True)
    df_results = pd.DataFrame({
        'experiment': np.repeat(experiment, len(cindex)),
        'model': np.repeat(model, len(cindex)),
        'iter': np.arange(0, len(cindex)),
        'cindex': cindex,
        'train_type': np.repeat(train_type, len(cindex))
    })
    result_file = os.path.join(results_dir, "results.csv")
    df_results.to_csv(result_file, mode='a', header=not os.path.exists(result_file))


class RollingMean:
    def __init__(self):
        self.vals = []

    def __str__(self):
        mean = self.get_mean()
        ci = self.get_ci()
        return f"{mean:{1}.{3}}, [{ci[0]:{1}.{3}} - {ci[1]:{1}.{3}}]"

    def add(self, val):
        self.vals.append(val)

    def get_mean(self):
        return np.mean(self.vals)

    def get_ci(self):
        mean, sigma = np.mean(self.vals), np.std(self.vals)
        conf_int = sp.stats.norm.interval(0.95, loc=mean, scale=sigma)
        return conf_int

    def get_max(self):
        return np.max(self.vals)


from sklearn.feature_selection import mutual_info_classif


def mutual_info_surv(X, y):
    y_np = np.asarray([[x[0], x[1]]for x in y]).astype(np.int)
    y_time_median = np.median(y_np[y_np[:, 0] == 1, 1])
    y_event_median = y_np[:, 1] < y_time_median
    return mutual_info_classif(X, y_event_median)
