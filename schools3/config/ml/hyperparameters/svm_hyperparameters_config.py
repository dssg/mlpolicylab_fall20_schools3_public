import sys
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()

def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_kernel_hp = \
    lambda x: get_hp(
        default='poly',
        val=x,
        is_categorical=True,
        categories=['linear', 'poly'],
    )

config.get_inv_reg_strength_hp = \
    lambda x: get_hp(
        default=0.05,
        val=x,
        hp_range=[1e-6, 100],
        log_scale=True,
        good_values=[0.05, 0.1, 0.3, 0.5, 0.7, 1]
    )

sys.modules[__name__] = config
