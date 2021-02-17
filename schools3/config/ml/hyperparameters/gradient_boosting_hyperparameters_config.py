import sys
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()

def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_learning_rate_hp = \
    lambda x: get_hp(
        default=0.03486,
        val=x,
        hp_range=[1e-6, 1],
        log_scale=True,
        good_values=[1E-4, 1E-2, 1E-1]
    )

config.get_n_estimators_hp = \
    lambda x: get_hp(
        default=600,
        val=x,
        hp_range=[1, 20000],
        good_values=[10000, 15000, 20000]
    )

config.get_max_depth_hp = \
    lambda x: get_hp(
        default=8,
        val=x,
        hp_range=[1, 100],
        good_values=[1, 3, 5, 10]
    )

sys.modules[__name__] = config
