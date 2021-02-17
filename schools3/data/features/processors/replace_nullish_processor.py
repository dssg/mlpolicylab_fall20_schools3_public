from schools3.data.features.processors.feature_processor import FeatureProcessor
import numpy as np

class ReplaceNullishProcessor(FeatureProcessor):
    def __init__(self, column_list:list=None, nullish=['none'], train_stats=None):
        '''
        Args:
            - column_dict: a list of column names to search through
        '''
        super(ReplaceNullishProcessor, self).__init__(train_stats)
        self.column_list = [] if column_list is None else column_list
        self.nullish = nullish

    def __call__(self, df, *args, **kwargs):
        new_df = df
        for c in self.column_list:
            for n in self.nullish:
                new_df[c] = df[c].replace(n, np.nan)

        return new_df
