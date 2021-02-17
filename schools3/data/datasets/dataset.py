from typing import List
import pandas as pd
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.labels.labels_table import LabelsTable
from schools3.data.base.cohort import Cohort
import schools3.config.data.datasets.dataset_config as config

# an dataset that a Model can train/predict on. Contains a cohort, features, and labels
class Dataset():
    def __init__(
        self, cohort:Cohort, features_list:List[FeaturesTable],
        final_feature_processors:CompositeFeatureProcessor, labels: LabelsTable
    ):
        self.cohort = cohort
        self.features_list = features_list
        self.final_feature_processor = final_feature_processors
        self.labels = labels
        self.proc_stats = None
        self.__dataset_df = self.join_features_labels()

    # returns the features and labels as two Dataframes
    def get_features_labels(self):
        features = self.__dataset_df.features
        labels = self.__dataset_df.labels

        if config.feat_whitelist:
            features = features[config.feat_whitelist]
        return features, labels

    # returns features and labels as a joined Dataframe
    def join_features_labels(self):
        # [id, year, grade] query
        labels_df = self.labels.get_for(self.cohort, keep_cohort_cols=True)

        features_df = self.cohort.get_cohort()
        cohort_cols = list(features_df.columns)
        for f in self.features_list:
            df = f.get_for(self.cohort, keep_cohort_cols=not f.high_school_features)
            features_df = features_df.merge(df, how='left', on=cohort_cols)

        labels_df = labels_df.set_index(cohort_cols)

        features_df = features_df.set_index(cohort_cols)
        features_df = features_df.loc[
            labels_df.index.intersection(features_df.index)
        ]
        features_df = self.final_feature_processor(features_df)
        self.proc_stats = self.final_feature_processor.get_stats()

        d = {}
        d['features'] = features_df
        d['labels'] = labels_df
        df = pd.concat(d, axis=1)

        return df

    # returns the entire dataset with cohort, features, and labels
    def get_dataset(self):
        return self.__dataset_df

    # return statistics generated during feature processing
    def get_feature_proc_stats(self):
        return self.proc_stats
