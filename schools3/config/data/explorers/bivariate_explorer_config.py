import sys
from schools3.config import base_config

config = base_config.Config()

config.pairplot_save_file = 'pairplots.pdf'

sys.modules[__name__] = config
