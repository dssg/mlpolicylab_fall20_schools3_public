import sys
from schools3.config import base_config

config = base_config.Config()

config.prefix = 'cat_'

sys.modules[__name__] = config
