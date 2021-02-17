from enum import Enum, auto
from typing import Union, Set
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from schools3.data.features.processors.feature_processor import FeatureProcessor
import schools3.config.data.features.processors.categorical_feature_processor_config as categorical_config
import schools3.config.data.features.processors.impute_null_processor_config as config

class ImputeBy(Enum):
    MEAN = pd.Series.mean
    MEDIAN = pd.Series.median


class ImputeNullProcessor(FeatureProcessor):
    def __init__(
        self, col_val_dict: dict = None,
        col_flag_set: Union[Set, bool] = True, fill_unspecified=None,
        flag_col_postfix=config.flag_col_postfix, last_resort_const=0,
        train_stats=None
    ):

        '''
        Args:
        - constant: a scalar value that will be used to replace all None values
            in the dataframe
        - col_val_dict: a dictionary of column names -> lambda | ImputeBy | value.
            Specifies how nulls should be replaced in each column.
        - col_flag_set: a set of column names that should have another column added to
            the dataframe to specify that they have been imputed or a boolean
            to decide whether all columns should have an extra column added
        - fill_unspecified: if not None, all columns not in col_val_dict will have their
            null values filled in by the function or constant passed in for
            `fill_unspecified`. If False, all columns not in col_val_dict will
            be not be imputed
        '''
        super(ImputeNullProcessor, self).__init__(train_stats)
        self.col_val_dict = {} if col_val_dict is None else col_val_dict
        self.col_flag_set = col_flag_set
        self.should_col_flag_set = \
            lambda c : self.col_flag_set if isinstance(self.col_flag_set, bool) else c in self.col_flag_set
        self.fill_unspecified = fill_unspecified
        self.flag_col_postfix = flag_col_postfix
        self.last_resort_const = last_resort_const

    def resolve_filler(self, filler, col):
        ans = None
        try:
            if callable(filler):
                ans = filler(col)
            elif isinstance(filler, ImputeBy):
                ans = filler(col)
            else:  # assume filler is scalar
                ans = filler
        except:
            return self.last_resort_const

        if ans is None or (isinstance(ans, float) and np.isnan(ans)):
            return self.last_resort_const

        return ans

    def add_null_indicators(self, df):
        for c in df.columns:
            if (self.fill_unspecified is not None) or (c in self.col_val_dict):
                if self.should_col_flag_set(c):
                    new_col = categorical_config.prefix + c + self.flag_col_postfix
                    assert new_col not in df.columns, 'choose different flag_col_postfix'
                    df[new_col] = df[c].isna().astype(CategoricalDtype([0, 1]))

        return df

    def __call__(self, df, *args, **kwargs):
        unspecified_cols = df.columns.difference(self.col_val_dict.keys())

        df = self.add_null_indicators(df)

        for c in self.col_val_dict:
            df[c] = df[c].fillna(
                        self.resolve_filler(self.col_val_dict[c], df[c])
                    )

        if self.fill_unspecified is not None:
            df[unspecified_cols] = df[unspecified_cols].fillna(df.apply(lambda col: self.resolve_filler(self.fill_unspecified, col)))

        return df
