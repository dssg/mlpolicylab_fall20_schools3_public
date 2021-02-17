import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from schools3.config.ml.metrics import performance_metrics_config

# base class for metrics, which are computed based on the predictions of a Model
class Metrics():
    def __init__(
        self,
        k=performance_metrics_config.best_precision_percent
    ):
        self.k = k

    # the "main" method of the Metrics class
    def compute(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    # create Dataframe with scores and labels as columns given their values
    def get_score_df(self, scores, labels):
        df = pd.DataFrame(np.array([scores, labels]).T)
        return df.rename(columns={0: 'score', 1: 'label'})

    # utility function for creating column names
    def get_metric_name(self, name):
        return f'{name} using top {round(self.k * 100, 2)}%'

    # given a score Dataframe, returns predictions and labels where top k% is positive
    def get_top_k(self, score_df: pd.DataFrame):
        # sort dataframe by descending score
        sorted_df = score_df.sort_values('score', ascending=False)

        # predict 1 on the top `self.k` percent
        selected_num    = int(sorted_df.shape[0] * self.k)
        labels          = sorted_df.label.values
        preds           = np.zeros(len(sorted_df))
        preds[:selected_num] = 1

        return preds, labels

    def compute_recall(self, labels, preds):
        metric_name = self.get_metric_name('recall')
        metric_val = recall_score(labels, preds, average='binary')

        return metric_name, metric_val

    # computes (true/false) (positive/negatives)
    def compute_confusion_matrix(self, labels, preds):
        vals = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        return dict(zip(['TN', 'FP', 'FN', 'TP'], vals))