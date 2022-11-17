from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sksurv.datasets import get_x_y
from sksurv.ensemble import RandomSurvivalForest
from skopt.space import Real, Categorical, Integer
import utils


def preprocess_data(dataset, config):
    X = dataset[config['features']]
    _, y = get_x_y(dataset,
                   attr_labels=[config['outcome']['event'], config['outcome']['time']],
                   pos_label=config['outcome']['pos_label'])
    return X, y


def train_model(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', RandomSurvivalForest())
        ]
    )

    parameters = {
        'model__n_estimators': Integer(5, 25),
        'model__max_depth': Categorical([3, 5, 7, 9]),
        'selector__k': Integer(1, X_train.shape[1]),
    }

    fit_params = {}

    return pipeline, parameters, fit_params
