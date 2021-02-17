from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.k_neighbors_hyperparameters_config as config

class KNeighborsHyperparameters(Hyperparameters):
    def __init__(self, n_neighbors=None):
        super(KNeighborsHyperparameters, self).__init__()
        self.n_neighbors = config.get_n_neighbors_hp(n_neighbors)
