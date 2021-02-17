import sys
import os
from schools3.config import base_config

config = base_config.Config()

config.min_test_rows = 200
config.get_model_csv_cache = lambda m: f'{m}_cache.csv'

def get_hash_pkl_file(model_name, hash_code):
    return f'saved_models/{model_name}_{hash_code}.pkl'

config.get_hash_pkl_file = get_hash_pkl_file

lo_data_districts = [
    'cat_district_Zanesville',
    'cat_district_New Lexington City SD',
    'cat_district_Logan Hocking',
    'cat_district_Morgan',
    'cat_district_Northern Local SD'
]

hi_data_districts = [
    'cat_district_West Muskingum',
    'cat_district_TriValley',
    'cat_district_Ridgewood',
    'cat_district_Riverview',
    'cat_district_Maysville',
    'cat_district_Franklin',
    'cat_district_East Muskingum',
    'cat_district_Coshocton',
    'cat_district_Crooksville'
]

config.ref_cols = {
    'ethnicity': 'cat_ethnicity_W',
    'district': (lo_data_districts, hi_data_districts)
}

sys.modules[__name__] = config
