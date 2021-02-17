import sys
from schools3.config import base_config

config = base_config.Config()
config.num_feats = list(range(10, 26, 5))

sys.modules[__name__] = config
