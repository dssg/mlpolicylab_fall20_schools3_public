import sys
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()


def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_max_iters_hp = \
    lambda x: get_hp(
        default=100,
        val=x,
        hp_range=[0, 10000],
        good_values=[100, 200, 300]
    )

config.get_penalty_hp = \
    lambda x: get_hp(
        default='elasticnet',
        val=x,
        is_categorical=True,
        categories=['elasticnet']
    )

config.get_l1_ratio_hp = \
    lambda x: get_hp(
        default=0.3,
        val=x,
        hp_range=[0, 1],
        good_values=[0, 0.1, 0.2, 0.3, 0.4]
    )

config.get_inv_reg_strength_hp = \
    lambda x: get_hp(
        default=0.01,
        val=x,
        hp_range=[0, 100],
        good_values=[0.01, 0.05, 0.1],
    )

config.get_seed_hp = \
    lambda x: get_hp(
        default=666666,
        val=x,
        hp_range=[0, 1000000],
        good_values=[666666],
    )

sys.modules[__name__] = config
