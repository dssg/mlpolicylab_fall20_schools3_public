import tensorflow as tf
from tensorflow.keras.regularizers import l2
from schools3.ml.models.tfkeras_model import TFKerasModel
from schools3.ml.hyperparameters.mlp_hyperparameters import MLPHyperparameters


class MLPModel(TFKerasModel):
    def __init__(self, hps:MLPHyperparameters=None):
        if hps is None:
            hps = MLPHyperparameters()

        layers = []
        for h in hps.hidden_sizes:
            layers.append(
                tf.keras.layers.Dense(
                        h,
                        activation='relu',
                        kernel_regularizer=l2(hps.reg_const)
                    )
            )

        model = tf.keras.Sequential(
            layers +
            [tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_regularizer=l2(hps.reg_const)
            )]
        )

        super(MLPModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return MLPHyperparameters
