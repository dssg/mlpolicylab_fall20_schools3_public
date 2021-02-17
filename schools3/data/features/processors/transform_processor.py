from enum import Enum
import numpy as np
from schools3.data.features.processors.feature_processor import FeatureProcessor


class Transforms(Enum):
    LOG = lambda x: np.log(x)
    SQRT = lambda x: np.sqrt(x)
    QUADRATIC = lambda x: np.power(x, 2)
    CUBIC = lambda x: np.power(x, 3)


class TransformProcessor(FeatureProcessor):
    def __init__(self, cols_dict, train_stats=None):
        self.cols_dict = cols_dict
        super(TransformProcessor, self).__init__(train_stats)

    def __call__(self, df, *args, **kwargs):
        for c in self.cols_dict:
            assert c in df.columns, f'column {c} does not exist in input dataframe'
            df[c] = df[c].apply(self.cols_dict[c])

        return df
