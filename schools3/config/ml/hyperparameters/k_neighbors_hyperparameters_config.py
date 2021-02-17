import sys
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()

def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_n_neighbors_hp = \
    lambda x: get_hp(
        default=5,
        val=x,
        hp_range=[1, 500],
        good_values=[1, 3, 5, 10, 50, 100, 500]
    )

sys.modules[__name__] = config
