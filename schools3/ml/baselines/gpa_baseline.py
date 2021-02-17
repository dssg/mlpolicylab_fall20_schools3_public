from schools3.ml.baselines.ranked_baseline import RankedBaseline
from schools3.data.features.academic_features import AcademicFeatures
import numpy as np
import pandas as pd


class GPABaseline(RankedBaseline):
    def __init__(self):
        features = AcademicFeatures()
        features_col = 'hs_avg_gpa'
        super(GPABaseline, self).__init__(
            features,
            features_col,
            is_sort_asc=True
        )
