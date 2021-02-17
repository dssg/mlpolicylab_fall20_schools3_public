import sys
from schools3.config import base_config

config = base_config.Config()

config.out_csv = 'tuning.csv'
config.out_img = 'tuning.html'

sys.modules[__name__] = config
