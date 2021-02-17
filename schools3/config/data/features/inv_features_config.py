import sys
from schools3.config import base_config


config = base_config.Config()

config.categorical_columns      = []

sys.modules[__name__] = config
