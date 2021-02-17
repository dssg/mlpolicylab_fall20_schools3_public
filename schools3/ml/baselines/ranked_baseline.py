import numpy as np
from schools3.ml.baselines.baseline import Baseline
from schools3.config.ml.metrics import performance_metrics_config

# a type of baseline that scores rows based on the ranks for a particular column
class RankedBaseline(Baseline):
    def __init__(
        self, features, feature_col, is_sort_asc
    ):
        super(RankedBaseline, self).__init__()
        self.features = features
        self.feature_col = feature_col
        self.is_sort_asc = is_sort_asc

    # assigns the scores and returns a Dataframe with the ranked feature, score and label
    def get_scores(self, cohort, labels_df):
        cohort_df = cohort.get_cohort()
        scores_df = self.features.get_for(cohort)[list(cohort_df) + [self.feature_col]]

        merged_df = scores_df.merge(labels_df, on=list(cohort.get_cohort().columns))
        merged_df = merged_df.set_index(list(cohort.get_cohort().columns))

        merged_df.rename(columns={self.feature_col: 'score'}, inplace=True)
        merged_df = merged_df.sort_values(by='score', ascending=self.is_sort_asc)

        # make score between 0 and 1
        merged_df['score'] = [x for x in reversed(np.linspace(0, 1, len(merged_df['score'])))]

        return merged_df
