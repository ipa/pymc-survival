import pymc as pm
from pmsurv.models import weibull_linear
import numpy as np
import time


def main():
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
