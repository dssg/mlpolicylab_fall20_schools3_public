import sys
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()

def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_max_depth_hp = \
    lambda x: get_hp(
        default=10,
        val=x,
        hp_range=[1, 150],
        is_none_allowed=True,
        good_values=[3, 5, 10, 50, None]
    )

config.get_min_samples_split_hp = \
    lambda x: get_hp(
        default=2,
        val=x,
        hp_range=[0, 10],
        good_values=[2, 5, 10]
    )

config.get_min_samples_leaf_hp = \
    lambda x: get_hp(
        default=1,
        val=x,
        hp_range=[0, 10],
        good_values=[1, 2, 5, 10]
    )

config.get_seed_hp = \
    lambda x: get_hp(
        default=666666,
        val=x,
        hp_range=[0, 1000000],
        good_values=[666666],
    )

sys.modules[__name__] = config
