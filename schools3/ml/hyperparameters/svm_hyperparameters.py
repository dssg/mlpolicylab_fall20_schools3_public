from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.svm_hyperparameters_config as config

class SVMHyperparameters(Hyperparameters):
    def __init__(self, kernel=None, reg_strength=None):
        super(SVMHyperparameters, self).__init__()
        self.kernel = config.get_kernel_hp(kernel)
        self.inv_reg_strength = config.get_inv_reg_strength_hp(reg_strength)
