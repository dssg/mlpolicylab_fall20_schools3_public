import pandas as pd
from pandas.api.types import CategoricalDtype
from schools3.data.features.processors.feature_processor import FeatureProcessor
import schools3.config.data.features.processors.categorical_feature_processor_config as config


class CategoricalFeatureProcessor(FeatureProcessor):
    def __init__(self, column_list:list=None, dummy_na=True, train_stats=None):
        '''
        Args:
        - column_list: set of column names to treat as categorical
        - dummy_na: if None/NaN should be treated as its own category
        '''
        super(CategoricalFeatureProcessor, self).__init__(train_stats)
        self.column_list = [] if column_list is None else column_list
        self.dummy_na = dummy_na

    def __call__(self, df, *args, **kwargs):
        renaming = {x: config.prefix + x for x in self.column_list}
        df = df.rename(columns=renaming)

        original_cols = df.columns.copy()

        new_df = pd.get_dummies(
            df, columns=renaming.values(),
            dummy_na=self.dummy_na, dtype=int
        )
        new_cols = new_df.columns.difference(original_cols)

        for c in new_cols:
            new_df[c] = new_df[c].astype(CategoricalDtype([0, 1]))

        return new_df

    def get_categorical_feature_names(self, feature, feature_cols):
        return feature_cols[
            feature_cols.str.startswith(config.prefix + feature + '_') &
            (~feature_cols.str.lstrip(config.prefix + feature + '_').str.contains('_'))
        ]
