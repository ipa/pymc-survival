import numpy as np
import lifelines
import lifelines.statistics
from lifelines import KaplanMeierFitter, plotting
# # import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


# def plot_survival_function(surv_prob_median, surv_prob_lower=None, surv_prob_upper=None, label='', max_time=50,
#                            median_threshold=0.5):
#     steps = surv_prob_median.shape[0]
#     t = np.linspace(1, max_time, steps)
#     median_ix = np.argmax(surv_prob_median <= median_threshold)
#     median_surv_prob = t[median_ix] if t[median_ix] > 1 else max_time + 1
#
#     sns.lineplot(t, surv_prob_median, label=label)
#     if surv_prob_lower is not None:
#         plt.fill_between(t, surv_prob_lower, surv_prob_upper, alpha=0.2)
#
#     if median_ix > 1:
#         plt.hlines(y=median_threshold, xmin=0, xmax=t[median_ix], color='black', linestyles='--')
#         plt.vlines(x=t[median_ix], ymin=0, ymax=median_threshold, color='black', linestyles='--')
#     return t, median_surv_prob


def compare_survival(data, group_col, time_col, event_col, xticks, labels):
    ix = data[group_col].values
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(data[time_col][ix], data[event_col][ix], label=labels[0])
    ax = kmf_low.plot_survival_function(show_censors=True)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(data[time_col][~ix], data[event_col][~ix], label=labels[1])
    ax = kmf_high.plot_survival_function(show_censors=True, ax=ax);
    plotting.add_at_risk_counts(kmf_low, kmf_high, rows_to_show=['At risk'], xticks=xticks, ax=ax)
    log_rank_results = lifelines.statistics.logrank_test(data[time_col][ix], data[time_col][~ix],
                                                         data[event_col][ix], data[event_col][~ix])
    p_val_txt = 'p = %.3f' % log_rank_results.p_value if log_rank_results.p_value >= 0.005 else 'p < 0.005'
    ax.add_artist(AnchoredText(p_val_txt, loc=4, frameon=False))
    ax.set_xticks(xticks)
    ax.set_ylim(0, 1.05)
    return ax


# '''
# Utility functions for running DeepSurv experiments
# '''
#
# import h5py
# import scipy.stats as st
# from collections import defaultdict
# import numpy as np
# import pandas as pd
# import copy
#
#
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
#
#
# def bootstrap_metric(metric_fxn, dataset, N=100):
#     def sample_dataset(dataset, sample_idx):
#         sampled_dataset = {}
#         for (key, value) in dataset.items():
#             sampled_dataset[key] = value[sample_idx]
#         return sampled_dataset
#
#     metrics = []
#     size = len(dataset['x'])
#
#     for _ in range(N):
#         resample_idx = np.random.choice(size, size=size, replace=True)
#
#         metric = metric_fxn(**sample_dataset(dataset, resample_idx))
#         metrics.append(metric)
#
#     # Find mean and 95% confidence interval
#     mean = np.mean(metrics)
#     conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
#     return {
#         'mean': mean,
#         'confidence_interval': conf_interval
#     }
#
#
# def calculate_recs_and_antirecs(rec_trt, true_trt, dataset, print_metrics=True):
#     if isinstance(true_trt, int):
#         true_trt = dataset['x'][:, true_trt]
#     rec_trt = np.asarray(rec_trt)
#     true_trt = np.asarray(true_trt)
#
#     # print(rec_trt, true_trt)
#     # trt_values = zip([0,1],np.sort(np.unique(true_trt)))
#     # trt_values = enumerate(np.sort(np.unique(true_trt)))
#     # equal_trt = [np.logical_and(rec_trt == rec_value, true_trt == true_value) for (rec_value, true_value) in trt_values]
#     rec_idx = rec_trt == true_trt
#     # print(rec_idx)
#     # rec_idx = np.logical_or(*equal_trt)
#     # original Logic
#     # rec_idx = np.logical_or(np.logical_and(rec_trt == 1,true_trt == 1),
#     #               np.logical_and(rec_trt == 0,true_trt == 0))
#
#     rec_t = dataset['t'][rec_idx]
#     antirec_t = dataset['t'][~rec_idx]
#     rec_e = dataset['e'][rec_idx]
#     antirec_e = dataset['e'][~rec_idx]
#
#     if print_metrics:
#         print("Printing treatment recommendation metrics")
#         metrics = {
#             'rec_median': np.median(rec_t),
#             'antirec_median': np.median(antirec_t)
#         }
#         print("Recommendation metrics:", metrics)
#
#     return {
#         'rec_t': rec_t,
#         'rec_e': rec_e,
#         'antirec_t': antirec_t,
#         'antirec_e': antirec_e
#     }


def get_time_axis(start, end, resolution):
    return np.linspace(start, end, int((end - start) * resolution + 1))

