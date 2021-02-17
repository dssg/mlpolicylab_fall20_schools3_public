import sys
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()

def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_n_estimators_hp = \
    lambda x: get_hp(
        default=2500,
        val=x,
        hp_range=[1, 15000],
        good_values=[2500, 5000, 7500, 10000, 12500]
    )

config.get_max_depth_hp = \
    lambda x: get_hp(
        default=100,
        val=x,
        hp_range=[1, 150],
        is_none_allowed=True,
        good_values=[75, 100, 125]
    )

config.get_max_features_hp = \
    lambda x: get_hp(
        default='log2',
        val=x,
        is_categorical=True,
        is_none_allowed=True,
        good_values=[None, 'sqrt', 'log2']
    )

config.get_min_samples_leaf_hp = \
    lambda x: get_hp(
        default=0.3,
        val=x,
        hp_range=[0, 1],
        good_values=[0.2, 0.3, 0.4]
    )

config.get_seed_hp = \
    lambda x: get_hp(
        default=666666,
        val=x,
        hp_range=[0, 1000000],
        good_values=[666666],
    )

sys.modules[__name__] = config
