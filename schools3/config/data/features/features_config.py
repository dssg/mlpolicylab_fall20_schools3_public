import sys
from schools3.config import base_config

config = base_config.Config()

config.min_grade = 6

sys.modules[__name__] = config
