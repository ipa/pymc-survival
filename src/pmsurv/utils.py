import numpy as np
import lifelines
import lifelines.statistics
from lifelines import KaplanMeierFitter, plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def plot_survival_function(surv_prob_median, surv_prob_lower=None, surv_prob_upper=None, label='', max_time=50,
                           median_threshold=0.5):
    steps = surv_prob_median.shape[0]
    t = np.linspace(1, max_time, steps)
    median_ix = np.argmax(surv_prob_median <= median_threshold)
    median_surv_prob = t[median_ix] if t[median_ix] > 1 else max_time + 1

    sns.lineplot(t, surv_prob_median, label=label)
    if surv_prob_lower is not None:
        plt.fill_between(t, surv_prob_lower, surv_prob_upper, alpha=0.2)

    if median_ix > 1:
        plt.hlines(y=median_threshold, xmin=0, xmax=t[median_ix], color='black', linestyles='--')
        plt.vlines(x=t[median_ix], ymin=0, ymax=median_threshold, color='black', linestyles='--')
    return t, median_surv_prob


def compare_survival(data, group_col, time_col, event_col, xticks, labels):
    ix = data[group_col].values
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(data[time_col][ix], data[event_col][ix], label=labels[0])
    ax = kmf_low.plot_survival_function(show_censors=True)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(data[time_col][~ix], data[event_col][~ix], label=labels[1])
    ax = kmf_high.plot_survival_function(show_censors=True, ax=ax)
    plotting.add_at_risk_counts(kmf_low, kmf_high, rows_to_show=['At risk'], xticks=xticks, ax=ax)
    log_rank_results = lifelines.statistics.logrank_test(data[time_col][ix], data[time_col][~ix],
                                                         data[event_col][ix], data[event_col][~ix])
    p_val_txt = 'p = %.3f' % log_rank_results.p_value if log_rank_results.p_value >= 0.005 else 'p < 0.005'
    ax.add_artist(AnchoredText(p_val_txt, loc=4, frameon=False))
    ax.set_xticks(xticks)
    ax.set_ylim(0, 1.05)
    return ax


def get_time_axis(start, end, resolution):
    return np.linspace(start, end, int((end - start) * resolution + 1))
