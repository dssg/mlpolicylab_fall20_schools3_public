import numpy as np
import sys
from schools3.config import base_config
from schools3.data.features.processors.impute_null_processor import ImputeBy

config= base_config.Config()

config.fill_values ={
    'absence_rate': ImputeBy.MEAN,
    'unexcused_absence_rate': ImputeBy.MEAN,
    'absence_rate_perc': ImputeBy.MEAN
}

sys.modules[__name__] = config
