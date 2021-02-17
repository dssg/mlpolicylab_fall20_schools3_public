from schools3.data.features.processors.feature_processor import FeatureProcessor


class PivotProcessor(FeatureProcessor):
    def __init__(self, index, columns, values, cols_postfix='', train_stats=None):
        super(PivotProcessor, self).__init__(train_stats)
        self.index = index
        self.columns = columns
        self.values = values
        self.cols_postfix = cols_postfix

    def __call__(self, df, *args, **kwargs):
        df = df.pivot(self.index, self.columns, self.values)
        df = df.rename(
            columns={c:str(c) + self.cols_postfix for c in df.columns}
        ).reset_index()
        df.columns.name = ''
        return df
