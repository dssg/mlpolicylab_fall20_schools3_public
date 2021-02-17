import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc

from schools3.ml.base.metrics import Metrics
from schools3.config.ml.metrics import performance_metrics_config

# a Metrics that computes fairness metrics such as F1, precision, AUC, etc for the top k%
class PerformanceMetrics(Metrics):
    def compute(self, scores_df):
        """
        params:
        - scores_df: [id, year, grade, score, label] pandas df

        ret: dict[string => float]. Maps metric names to values
        """
        metrics = {}

        preds, labels = self.get_top_k(scores_df)

        # compute metrics on all rows
        for metric_func in [self.compute_accuracy, self.compute_auc, self.compute_f1, self.compute_recall,
                            self.compute_precision]:
            metric_name, metric_val = metric_func(labels, preds)
            metrics[metric_name] = metric_val

        return pd.DataFrame({k: [metrics[k]] for k in metrics})

    def compute_f1(self, labels, preds):
        metric_name = self.get_metric_name('f1-score')
        metric_val = f1_score(labels, preds)

        return metric_name, metric_val

    def compute_precision(self, labels, preds):
        metric_name = self.get_metric_name('precision')
        metric_val  = precision_score(labels, preds, average='binary')

        return metric_name, metric_val

    def compute_accuracy(self, labels, preds):
        metric_name = self.get_metric_name('accuracy')
        metric_val = accuracy_score(labels, preds)

        return metric_name, metric_val

    def compute_auc(self, labels, preds):
        metric_name = self.get_metric_name('auc')
        fpr, tpr, _ = roc_curve(labels, preds)
        metric_val  = auc(fpr, tpr)

        return metric_name, metric_val
