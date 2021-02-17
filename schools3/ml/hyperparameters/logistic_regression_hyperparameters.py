from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.logistic_regression_hyperparameters_config as config

class LogisticRegressionHyperparameters(Hyperparameters):
    def __init__(self, penalty=None, max_iters=None, reg_strength=None, l1_ratio=None, seed=None):
        super(LogisticRegressionHyperparameters, self).__init__()
        self.penalty = config.get_penalty_hp(penalty)
        self.max_iters = config.get_max_iters_hp(max_iters)
        self.inv_reg_strength = config.get_inv_reg_strength_hp(reg_strength)
        self.l1_ratio = config.get_l1_ratio_hp(l1_ratio)
        self.seed = config.get_seed_hp(seed)
