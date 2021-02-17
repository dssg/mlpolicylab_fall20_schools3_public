import sys
import os
import pandas as pd
from schools3.config import base_config
from schools3.config import global_config


config = base_config.Config()

config.null_replacement = 'NA'
COL_INV_NAME, COL_INV_DES, COL_INV_LABEL = 'category', 'description', 'Label-ActiveGood'

def load_mapping(col_label):
    file_mapping    = os.path.join(global_config.etc_invfeat_dir, 'inv-feature-mapping.csv')
    df_mapping      = pd.read_csv(file_mapping).loc[:, [COL_INV_NAME, COL_INV_DES, col_label]]
    df_filter       = df_mapping[df_mapping.loc[:, col_label] != 'Positive']

    df_filter[COL_INV_NAME] = df_filter[COL_INV_NAME].fillna(config.null_replacement)

    dict_mapping = {}
    for idx in range (df_filter.shape[0]):
        category, description, label = list(df_filter.iloc[idx, :].values)

        ## initialize
        if category not in dict_mapping:
            dict_mapping[category] = [[], []]

        if label == 'Neutral':
            dict_mapping[category][0].append(description)
        elif label == 'Negative':
            dict_mapping[category][1].append(description)

    return dict_mapping

### manual mapping
config.mapping = load_mapping()

sys.modules[__name__] = config
