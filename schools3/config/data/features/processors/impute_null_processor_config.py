import sys
from schools3.config import base_config

config = base_config.Config()

config.flag_col_postfix = '_imputed'

sys.modules[__name__] = config
