import sys
from schools3.config import base_config

config = base_config.Config()

config.n_features   = 5
config.top_labels   = 1
config.n_analyze    = 3

config.class_names  = ['no_risk', 'high_risk']
config.class_name   = ['high_risk']

sys.modules[__name__] = config
