import pandas as pd
from typing import List
from schools3.ml.base.metrics import Metrics

# a Metrics that allows for combining multiple other Metrics
class CompositeMetrics(Metrics):
    def __init__(self, metrics=List[Metrics]):
        self.metrics = metrics

    def compute(self, scores_df):
        dfs = []
        for m in self.metrics:
            dfs.append(m.compute(scores_df))

        return pd.concat(
            dfs,
            axis=1,
            join='outer'
        )
