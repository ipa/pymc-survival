import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pmsurv.utils import get_time_axis, plot_survival_function, compare_survival


def test_get_time_axis_length():
    start, end, resolution = 0, 10, 2
    t = get_time_axis(start, end, resolution)
    expected_len = int((end - start) * resolution + 1)
    assert len(t) == expected_len
    assert t[0] == start and t[-1] == end


def test_plot_survival_function_no_threshold_cross():
    surv_prob_median = np.ones(10)  # never falls below 0.5
    t, median_surv_prob = plot_survival_function(surv_prob_median, median_threshold=0.5, max_time=50)
    # per implementation, when no crossing occurs median_surv_prob == max_time + 1
    assert median_surv_prob == 51
    plt.close('all')


def test_compare_survival_boolean_group():
    # create small dataset
    df = pd.DataFrame({
        'group': [True, True, False, False, True],
        'time': [1,2,3,4,5],
        'event': [1,0,1,0,1]
    })
    ax = compare_survival(df, group_col='group', time_col='time', event_col='event', xticks=[0,2,4], labels=['A','B'])
    assert ax is not None
    plt.close('all')
