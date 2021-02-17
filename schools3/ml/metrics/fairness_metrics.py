import pandas as pd
from schools3.ml.base.metrics import Metrics
import numpy as np

# a Metrics that computes fairness metrics such as FOR, FNR and FDR for the top k%
class FairnessMetrics(Metrics):
    def compute(self, scores_df):
        """
        params:
        - scores_df: [id, year, grade, score, label] pandas df

        ret: dict[string => float]. Maps metric names to values
        """
        metrics = {}

        preds, labels = self.get_top_k(scores_df)
        metric_dict = self.compute_confusion_matrix(labels, preds)

        # compute metrics on all rows
        for metric_func in [self.compute_fnr]:
            metric_name, metric_val = metric_func(metric_dict)
            metrics[metric_name] = metric_val

        return pd.DataFrame({k: [metrics[k]] for k in metrics})

    def compute_for(self, metric_dict):
        metric_name = self.get_metric_name('for')

        pred_neg = metric_dict['FN'] + metric_dict['TN']
        if pred_neg == 0:
            return metric_name, np.nan

        return metric_name, metric_dict['FN'] / pred_neg

    def compute_fnr(self, metric_dict):
        metric_name = self.get_metric_name('fnr')
        pos = metric_dict['FN'] + metric_dict['TP']
        if pos == 0:
            return metric_name, np.nan

        return metric_name, metric_dict['FN'] / pos

    def compute_fdr(self, metric_dict):
        metric_name = self.get_metric_name('fdr')
        pos = metric_dict['FP'] + metric_dict['TP']
        if pos == 0:
            return metric_name, np.nan

        return metric_name, metric_dict['FP'] / pos

    def compute_tpr(self, metric_dict):
        _, fnr = self.compute_fnr(metric_dict)
        return self.get_metric_name('tpr'), 1 - fnr
