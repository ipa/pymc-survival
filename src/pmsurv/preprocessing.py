import warnings
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import copy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class LogStandardScaler:
    def __init__(self, selected_columns):
        super().__init__()
        self.selected_columns = selected_columns
        self.pipe = Pipeline([
            ('log1p', preprocessing.FunctionTransformer(np.log1p, validate=True)),
            ('scaler', preprocessing.StandardScaler())
        ])

    def fit(self, x):
        x_ = copy.deepcopy(x)
        self.pipe.fit(x_[:, self.selected_columns])

    def transform(self, x):
        x_ = copy.deepcopy(x)
        x_[:, self.selected_columns] = self.pipe.transform(x_[:, self.selected_columns])
        return x_

    def reverse(self, x):
        x_ = copy.deepcopy(x)
        x_[:, self.selected_columns] = self.pipe.inverse_transform(x_[:, self.selected_columns])
        return x_


class SelectedColumnsRobustScaler:
    def __init__(self, selected_columns):
        super().__init__()
        self.selected_columns = selected_columns
        self.pipe = Pipeline([
            ('scaler', preprocessing.RobustScaler())])

    def fit(self, x):
        x_ = copy.deepcopy(x)
        self.pipe.fit(x_[:, self.selected_columns])

    def transform(self, x):
        x_ = copy.deepcopy(x)
        x_[:, self.selected_columns] = self.pipe.transform(x_[:, self.selected_columns])
        return x_
