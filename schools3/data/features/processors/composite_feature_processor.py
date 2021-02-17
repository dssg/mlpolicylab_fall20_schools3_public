from schools3.data.features.processors.feature_processor import FeatureProcessor
from typing import List


class CompositeFeatureProcessor(FeatureProcessor):
    def __init__(self, feature_processors: List[FeatureProcessor], train_stats=None):
        super(CompositeFeatureProcessor, self).__init__(train_stats)
        self.processors = feature_processors

    def __call__(self, df, *args, **kwargs):
        new_df = df
        for processor in self.processors:
            new_df = processor(new_df)

        return new_df

    def get_stats(self):
        ans = []
        for p in self.processors:
            ans.append(p.get_stats())
        return ans
