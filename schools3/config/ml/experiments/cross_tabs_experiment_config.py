import sys
from schools3.config import base_config

config = base_config.Config()

config.num_features = 10

sys.modules[__name__] = config
