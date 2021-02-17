import sys
from schools3.config import base_config
from schools3.data.features.processors.impute_null_processor import ImputeBy

config = base_config.Config()

config.fill_values = {
    'limited_english': 'N',
    'discipline_incidents': 0,
    'pre_1_year_discipline_incidents': 0,
    'pre_2_year_discipline_incidents': 0,
    'cumul_discipline_incidents': 0,
    'days_absent': 0,
    'pre_1_year_days_absent': 0,
    'pre_2_year_days_absent': 0,
    'num_transfers': 1,
}

config.impute_flag_columns = True

config.categorical_columns = [
    'gender',
    'ethnicity',
    'school_name',
    'district',
    'disability',
    'disadvantagement',
    'limited_english',
]

config.replace_nullish_columns = [
    'disability',
    'disadvantagement',
]

sys.modules[__name__] = config
