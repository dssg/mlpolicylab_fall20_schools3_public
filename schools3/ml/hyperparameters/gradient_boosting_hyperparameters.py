from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.gradient_boosting_hyperparameters_config as config

class GradientBoostingHyperparameters(Hyperparameters):
    def __init__(self, learning_rate=None, n_estimators=None, max_depth=None):
        super(GradientBoostingHyperparameters, self).__init__()
        self.learning_rate = config.get_learning_rate_hp(learning_rate)
        self.n_estimators = config.get_n_estimators_hp(n_estimators)
        self.max_depth = config.get_max_depth_hp(max_depth)
