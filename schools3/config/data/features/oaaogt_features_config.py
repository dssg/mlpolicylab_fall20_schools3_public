import numpy as np
import sys
from schools3.config import base_config

config = base_config.Config()

config.categorical_columns = [
    'sixth_read_pl',
    'sixth_math_pl',
    'sixth_write_pl',
    'sixth_ctz_pl',
    'sixth_science_pl',
    'seventh_read_pl',
    'seventh_math_pl',
    'seventh_write_pl',
    'eighth_read_pl',
    'eighth_math_pl',
    'eighth_science_pl',
    'eighth_socstudies_pl',
]

sys.modules[__name__] = config
