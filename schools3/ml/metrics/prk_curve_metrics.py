import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from schools3.ml.base.metrics import Metrics
import schools3.config.ml.metrics.prk_curve_metrics_config as config
import schools3.config.global_config as global_config
from sklearn.metrics import precision_score, recall_score

# a Metrics that computes fairness metrics such as precision and recall for different levels of k
class PRkCurveMetrics(Metrics):
    def __init__(self, file_name=config.save_file):
        super(PRkCurveMetrics, self).__init__()
        self.save_path = global_config.get_save_path(file_name)

    def compute(self, scores_df):
        sorted_df = scores_df.sort_values('score', ascending=False)

        metrics = {
            'k': [],
            'num_intervened': [],
            'precision': [],
            'recall': []
        }

        plotly_metrics = {
            'k': [],
            'value': [],
            'metric': [],
        }

        for k in range(config.k_step_size, 100 + config.k_step_size, config.k_step_size):
            metrics['k'].append(k)

            n = int(sorted_df.shape[0] * k / 100)
            preds = np.concatenate([np.ones(n), np.zeros(sorted_df.shape[0] - n)])
            precision = precision_score(sorted_df.label[:n].values, preds[:n])
            recall = recall_score(sorted_df.label.values, preds)

            metrics['num_intervened'].append(n)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)

            plotly_metrics['k'].append(k)
            plotly_metrics['value'].append(precision)
            plotly_metrics['metric'].append('precision')
            plotly_metrics['k'].append(k)
            plotly_metrics['value'].append(recall)
            plotly_metrics['metric'].append('recall')

        plot_df = pd.DataFrame({k: plotly_metrics[k] for k in plotly_metrics})
        fig = px.line(plot_df, x='k', y='value', color='metric')
        fig.write_html(self.save_path)

        df = pd.DataFrame({k: metrics[k] for k in metrics})
        df = df.sort_values('precision', ascending=False)

        return df
