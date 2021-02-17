from schools3.data.features.processors.feature_processor import FeatureProcessor
from schools3.config.data.features import inv_type_processor_config as config


class InvTypeProcessor(FeatureProcessor):
    def __init__(self, index, columns, values, cols_postfix=''):
        super(InvTypeProcessor, self).__init__()
        self.index = index
        self.columns = columns
        self.values = values
        self.cols_postfix = cols_postfix

    def __call__(self, df, *args, **kwargs):
        df['pivot_inv_group'].replace('', config.null_replacement, inplace=True)
        df = df.drop_duplicates(subset=self.index)

        for idx_col, col_label in enumerate(config.COL_INV_LABEL):
            mapping     = config.COL_MAPPING[col_label]
            col_name    = self.values[idx_col]

            for idx in range (df.shape[0]):
                inv_group, desc = list(df.iloc[idx][['pivot_inv_group', 'description']].values)
                if inv_group in mapping:
                    if desc in mapping[inv_group][0]: # Neutral
                        df[col_name].iloc[idx] = 0
                    elif desc in mapping[inv_group][1]: # Negative
                        df[col_name].iloc[idx] = -1
                    else:
                        df[col_name].iloc[idx] = 1 # Positive
                else:
                    df[col_name].iloc[idx] = 1 # Positive

        ret_df = df.drop(self.columns, axis=1)

        return ret_df
