from schools3.data.features.processors.feature_processor import FeatureProcessor
from schools3.config.data.features.processors import categorical_feature_processor_config


class StandardizeProcessor(FeatureProcessor):
    def __init__(self, post_fill_const=None, train_stats=None):
        '''
            Args:
                post_fill_const: A constant to impute all None values by after
                                standardizing. A useful constant to use could be
                                0 for some models
        '''
        super(StandardizeProcessor, self).__init__(train_stats)
        self.post_fill_const = post_fill_const
        if self.train_stats is None:
            self.mean, self.std = None, None
        else:
            self.mean, self.std = self.train_stats

    def __call__(self, df, eps=1e-4, *args, **kwargs):
        # FIXME: Returns NaN for non numeric columns

        standardizable_cols = []
        for c in df.columns.tolist():
            if c in ['student_lookup', 'school_year', 'grade'] or\
                c.startswith(categorical_feature_processor_config.prefix) or\
                df[c].dtype.name == 'category':
                continue

            standardizable_cols.append(c)

        if self.mean is None:
            self.mean = df[standardizable_cols].mean()
        if self.std is None:
            self.std  = df[standardizable_cols].std()

        df[standardizable_cols] = (df[standardizable_cols] - self.mean) / (self.std + eps)

        if self.post_fill_const is not None:
            df[standardizable_cols] = df[standardizable_cols].fillna(self.post_fill_const)

        return df

    def get_stats(self):
        return self.mean, self.std
