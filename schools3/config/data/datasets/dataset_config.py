import sys
from schools3.config import base_config

config = base_config.Config()

config.feat_whitelist = []

sys.modules[__name__] = config
