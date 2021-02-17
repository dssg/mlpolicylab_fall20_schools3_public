from schools3.ml.baselines.ranked_baseline import RankedBaseline
from schools3.data.features.absence_features import AbsenceFeatures
import numpy as np
import pandas as pd


class AbsenteeismBaseline(RankedBaseline):
    def __init__(self):
        features = AbsenceFeatures()
        features_col = 'absence_rate'
        super(AbsenteeismBaseline, self).__init__(
            features,
            features_col,
            is_sort_asc=False
        )
