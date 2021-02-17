import sys
from tensorflow.keras.optimizers import Adam, SGD
from schools3.config import base_config
from schools3.ml.base.hyperparameters import Hyperparameter

config = base_config.Config()


def get_hp(default, val=None, **kwargs):
    if val is None:
        val = default
    return Hyperparameter(val=val, **kwargs)

config.get_hidden_sizes_hp = \
    lambda x: get_hp(
        default=[64, 32, 16],
        val=x,
        is_categorical=True,
        categories=[
            [128, 64, 32],
            [128, 16],
            [128, 32],
            [64, 32, 16],
            [32, 16]
        ]
    )

config.get_optimizer_hp = \
    lambda x: get_hp(
        default=SGD(learning_rate=0.01, momentum=0.9),
        val=x,
        is_categorical=True,
        categories=[
            SGD(learning_rate=0.01, momentum=0.9),
            SGD(learning_rate=0.1, momentum=0.9),
            SGD(learning_rate=0.001, momentum=0.9)
        ],
        print_func=(lambda x: '_'.join(map(str, x.get_config().values())))
    )

config.get_loss_hp = \
    lambda x: get_hp(
        default='binary_crossentropy',
        val=x,
        is_categorical=True,
        categories=['binary_crossentropy']
    )

config.get_epochs_hp = \
    lambda x: get_hp(
        default=1000,
        val=x,
        hp_range=[1, 1000],
        good_values=[1000]
    )

config.get_batch_size_hp = \
    lambda x: get_hp(
        default=128,
        val=x,
        hp_range=[1, 256],
        good_values=[32, 64, 128, 256],
    )

config.get_reg_const_hp = \
    lambda x: get_hp(
        default=0.05,
        val=x,
        hp_range=[1e-6, 1],
        good_values=[0.01, 0.05, 0.1, 1],
    )

sys.modules[__name__] = config
