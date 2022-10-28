# import aesara
# aesara.compile.mode.Mode(linker='py', optimizer='fast_compile')
import pmsurv
import pmsurv.models
import pymc as pm
from pmsurv.models import weibull_linear
import numpy as np
import time
import pandas as pd


def main():
    data = pd.read_csv("data/veteran.csv")
    X = data[['age', 'celltype', 'trt']]
    X['celltype_1'] = data['celltype'] == 1
    X['celltype_2'] = data['celltype'] == 2
    X['celltype_3'] = data['celltype'] == 3
    X['celltype_4'] = data['celltype'] == 4
    X['trt'] = data['trt'] == 2

    y = data[['time', 'status']].values
    y[:, 1] = 1 - y[:, 1]  # inverse

    wb_model = weibull_linear.WeibullModelLinear()
    fit_args = {'draws': 1000, 'tune': 500, 'target_accept': 0.8, 'chains': 2, 'cores': 1,
                'return_inferencedata': True, 'progressbar': True}
    wb_model.fit(X, y, inference_args=fit_args)

    wb_model.save('data/model.yaml')

    model = weibull_linear.WeibullModelLinear()
    model.load('data/model.yaml')

    print(model.trace)

    x_predict = np.asarray([[2 * 10, 2, 3, 4, 0, 0, 0]])
    start = time.time()
    cached_model = model.create_model()
    end = time.time()
    print('create model: ', end - start)
    with cached_model:
        print('from cached model')
        pm.set_data({
            'model_input': x_predict,
            # 'time_uncensor': np.zeros(num_samples).astype(np.int32),
            # 'censor': np.zeros(num_samples).astype(np.int32)
        })
        print('data set')
        start = time.time()
        ppc = pm.sample_posterior_predictive(model.trace, return_inferencedata=False, progressbar=True,
                                             keep_size=False, samples=1000,
                                             random_seed=0, var_names=['lambda_det', 'k_det'])

        end = time.time()
        print('sample: ', end - start)

        print('run next sample')

        start = time.time()
        ppc = pm.sample_posterior_predictive(model.trace, return_inferencedata=False, progressbar=True,
                                             keep_size=False, samples=1000,
                                             random_seed=0, var_names=['lambda_det', 'k_det'])

        end = time.time()
        print('sample 2: ', end - start)

        print(ppc['lambda_det'].shape)

if __name__ == "__main__":
    main()