from schools3.data.features.processors.feature_processor import FeatureProcessor
from schools3.config.data.features.processors import oaaogt_processor_config


class OAAOGTProcessor(FeatureProcessor):
    def __init__(self, ignore_cols=['student_lookup', 'school_year', 'grade'], train_stats=None):
        self.ignore_cols = ignore_cols

    def get_ordinal_score(self, x):
        new_x = x.lower() if isinstance(x, str) else x
        return oaaogt_processor_config.pl_score.get(new_x, None)

    def __call__(self, df, *args, **kwargs):
        for c in df.columns:
            if c in self.ignore_cols:
                continue
            new_col = c + oaaogt_processor_config.ordinal_post_fix
            assert new_col not in df.columns
            df[new_col] = df[c].apply(self.get_ordinal_score)

        return df
