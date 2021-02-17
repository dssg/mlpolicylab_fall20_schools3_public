import sys
from schools3.config import base_config

config = base_config.Config()

config.low_gpa_percentile = 0.15
config.high_gpa_percentile = 0.5

config.dropout              = 'dropout'
config.late_grad            = 'late_grad'
config.unconfirmed_low_gpa  = 'unconfirmed_low_gpa'
config.on_time_grad         = 'on_time_grad'
config.unconfirmed_high_gpa = 'unconfirmed_high_gpa'
config.unclassified         = 'unclassified'


config.labels_dict = {
    'dropout': 1,
    'late_grad': 1,
    'unconfirmed_low_gpa': 1,
    'on_time_grad': 0,
    'unconfirmed_high_gpa': 0,
    'unclassified': 0
}

sys.modules[__name__] = config
