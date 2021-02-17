from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.random_forest_hyperparameters_config as config

class RandomForestHyperparameters(Hyperparameters):
    def __init__(self, n_estimators=None, max_depth=None, min_samples_leaf=None, seed=None, max_features=None):
        super(RandomForestHyperparameters, self).__init__()
        self.n_estimators = config.get_n_estimators_hp(n_estimators)
        self.max_depth = config.get_max_depth_hp(max_depth)
        self.min_samples_leaf = config.get_min_samples_leaf_hp(min_samples_leaf)
        self.seed = config.get_seed_hp(seed)
        self.max_features = config.get_max_features_hp(max_features)
