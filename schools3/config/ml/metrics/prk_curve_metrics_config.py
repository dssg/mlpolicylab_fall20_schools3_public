import sys
from datetime import datetime
from schools3.config import base_config

config = base_config.Config()

config.k_step_size = 1
config.save_file   = f'prk_curve_{datetime.now().strftime("%b_%d__%H_%M")}.html'

sys.modules[__name__] = config
