import sys
from schools3.config import base_config

config = base_config.Config()

### ratio/percentage of predictions to consider (e.g. 1.0: consider all predictions; 0.5 consider only top 50%
config.best_precision_percent = 0.05

sys.modules[__name__] = config
