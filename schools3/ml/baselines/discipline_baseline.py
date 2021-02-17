from schools3.ml.baselines.ranked_baseline import RankedBaseline
from schools3.data.features.snapshot_features import SnapshotFeatures
import numpy as np
import pandas as pd


class DisciplineBaseline(RankedBaseline):
    def __init__(self):
        features = SnapshotFeatures()
        features_col = 'discipline_incidents'
        super(DisciplineBaseline, self).__init__(
            features,
            features_col,
            is_sort_asc=False
        )
