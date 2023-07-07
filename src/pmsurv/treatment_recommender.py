import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import json


class TreatmentRecommender:
    def __init__(self, survival_model):
        self.survival_model = survival_model
        self.thresholds = {'low_risk': None}

    def save(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.thresholds, json_file)

    def load(self, file_name):
        with open(file_name, 'r') as json_file:
            self.thresholds = json.load(json_file)

    def find_best_threshold(self, x, y):
        surv_prob, _, _ = self.survival_model.predict(x)
        surv_prob_median = np.median(surv_prob, axis=1)

        event = y[:, 1] == 0
        median_risk = 1 - surv_prob_median
        fpr, tpr, thresholds = roc_curve(event, median_risk)
        auc = roc_auc_score(event, median_risk)
        fpri = 1 - fpr
        youden = (tpr + fpri - 1)
        ix_best = np.argmax(youden)
        ix_fpr = np.argwhere(fpr == np.min(fpr))[-1][0]
        ix_tpr = np.argmax(tpr)
        print('Best Threshold=%.3f, Youden=%.3f' % (thresholds[ix_best], youden[ix_best]))
        print('AUC %.3f' % (auc))
        print(ix_fpr, ix_tpr)

        best_thresh_median = np.median(median_risk)
        thresholds_dict = {'low_risk': best_thresh_median,
                           'best': thresholds[ix_best],
                           'best_tpr': thresholds[ix_tpr],
                           'best_fpr': thresholds[ix_fpr]
                           }
        return thresholds, thresholds_dict

    def fit(self, x, y):
        _, thresholds = self.find_best_threshold(x, y)
        self.thresholds = thresholds
        print(self.thresholds)

    def predict_risk(self, x, y=None, threshold_idx='low_risk'):
        surv_prob, _, _ = self.survival_model.predict(x)
        surv_prob_median = np.median(np.nan_to_num(surv_prob, 0), axis=1)

        median_risk = 1 - surv_prob_median
        risk = ["Low" if x < self.thresholds[threshold_idx] else "High" for x in median_risk]
        df_risk = pd.DataFrame({'time': 0 if y is None else y[:, 0],
                                'event': 0 if y is None else 1 - y[:, 1],
                                'risk': risk})
        return df_risk

    def predict(self, x, treatment_options, threshold_idx='low_risk', treatment_idx=-1):
        import copy
        options_label = list(range(len(treatment_options)))
        x_ = copy.deepcopy(x)
        df_recommend = pd.DataFrame({'actual_treatment': x_[:, treatment_idx],
                                     'recommended': max(treatment_options)})
        df_risk = pd.DataFrame(columns=['treatment', 'risk'])
        # print(treatment_options)
        n_samples = x_.shape[0]
        x_ = np.tile(x_, (len(treatment_options), 1))

        treatment_options_rep = np.repeat(treatment_options, n_samples)
        x_[:, treatment_idx] = treatment_options_rep
        surv_prob, _, _ = self.survival_model.predict(x_)

        for idx, treatment in enumerate(treatment_options):
            print(f"predict risk for treatment {treatment}")
            # print(treatment_options_rep)
            surv_treatment_idx = np.argwhere(treatment_options_rep == treatment).flatten()
            # print(surv_treatment_idx)
            surv_prob_treat = surv_prob[surv_treatment_idx, :]
            # print(surv_prob_treat)
            surv_prob_median = np.median(np.nan_to_num(surv_prob_treat, 0), axis=1)
            # print(surv_prob_median)
            median_risk = 1 - surv_prob_median
            df_recommend[str(options_label[idx])] = median_risk

            df_risk = df_risk.append({'treatment': np.repeat(treatment, len(median_risk)).astype(np.float64),
                                      'risk': np.asarray(median_risk).astype(np.float64)}, ignore_index=True)
            if idx >= 1:
                risk_class = np.array(
                    ["Low" if x < self.thresholds[threshold_idx] else "High" for x in median_risk])
                previous_risk_class = np.array(
                    ["Low" if x < self.thresholds[threshold_idx] else "High" for x in
                     df_recommend[str(options_label[idx - 1])]])
                ix = (previous_risk_class == "High") & (risk_class == "Low")
                df_recommend.loc[ix, ['recommended']] = treatment

        return df_recommend, df_risk
