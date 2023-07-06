import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


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

    X_train = X.values.astype(float)
    y_train = 1 - y[:, 1].astype(float)

    n_predictors = X.shape[1]

    print(X_train.shape)
    print(y_train.shape)

    with pm.Model() as model:  # noqa: F841
        ell = pm.InverseGamma("ell", mu=1.0, sigma=0.5, shape=(n_predictors,))
        eta = pm.Exponential("eta", lam=1.0)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(input_dim=n_predictors, ls=ell)

        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=X_train)

        # logit link and Bernoulli likelihood
        p = pm.Deterministic("p", pm.math.invlogit(f))
        # lambda_det = pm.Deterministic("lambda_det", pm.math.exp(f))
        y_ = pm.Bernoulli("y", p=p, observed=y_train)  # noqa: F841

        # censor_ = at.eq(censor_, 1)
        # y = pm.Exponential("y", at.ones_like(time_uncensor_) / lambda_det[~censor_],
        #                    observed=time_uncensor_)
        #
        # def exponential_lccdf(lam, time):
        #     """ Log complementary cdf of Exponential distribution. """
        #     return -(lam * time)
        #
        # y_cens = pm.Potential(
        #     "y_cens", exponential_lccdf(at.ones_like(time_censor_) / lambda_det[censor_], time_censor_)
        # )

        trace = pm.sample(draws=100, tune=50, chains=2, random_seed=1, cores=1, progressbar=True)

    az.plot_trace(trace)
    plt.show()


if __name__ == "__main__":
    main()
