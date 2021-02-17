import sys
from schools3.config import base_config
from sklearn.metrics import precision_score, roc_curve, auc, make_scorer
import numpy as np

config = base_config.Config()

config.tensorboard_log_dir = 'runs'
config.patience = 35

sys.modules[__name__] = config
