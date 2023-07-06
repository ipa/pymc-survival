import numpy as np
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score


def bootstrap_metric(metric_fxn, x, y, N=100):
    metrics = []
    size = x.shape[0]

    for _ in tqdm(range(N)):
        try:
            resample_idx = np.random.choice(size, size=size, replace=True)
            metric = metric_fxn(x[resample_idx, :], y[resample_idx, :])
            metrics.append(np.round(metric, 3))
        except ZeroDivisionError:
            pass

    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics)-1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }


def evaluate_model(model, x, y, bootstrap=False):
    def ci(model):
        def cph_ci(x, y, **kwargs):
            # max_time = np.max(y[:,0])
            # t = (np.ones((x.shape[0], 200)) * np.linspace(1, max_time, 200)).T
            return model.score(x, y)

        return cph_ci

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(x, y)
    if bootstrap:
        metrics['c_index_bootstrap'] = bootstrap_metric(ci(model), x, y)

    return metrics


def plot_roc_curve(surv_prob, y):
    surv_prob_median = np.median(surv_prob, axis=1)
    event = y[:, 1] == 0
    median_risk = 1 - surv_prob_median
    fpr, tpr, thresholds = roc_curve(event, median_risk)
    auc = roc_auc_score(event, median_risk)
    fpri = 1 - fpr
    youden = (tpr + fpri - 1)
    ix = np.argmax(youden)
    ix_fpr = np.argwhere(fpr == np.min(fpr))[-1]
    ix_tpr = np.argmax(tpr)
    ix_median = np.argwhere(thresholds <= np.median(median_risk))[0]
    print('Best Threshold=%.3f, Youden=%.3f' % (thresholds[ix], youden[ix]))
    print('AUC %.3f' % (auc))

    ax = plt.subplot(1, 1, 1)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                      estimator_name='ROC')
    display.plot(ax=ax)
    ax.plot((0, 1), (0, 1), ls="--", c=".3")
    ax.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Youden')
    ax.scatter(fpr[ix_fpr], tpr[ix_fpr], marker='o', color='green', label='Best FPR')
    ax.scatter(fpr[ix_tpr], tpr[ix_tpr], marker='o', color='red', label='Best TPR')
    ax.scatter(fpr[ix_median], tpr[ix_median], marker='x', color='orange', label='Best Median')
    return ax
