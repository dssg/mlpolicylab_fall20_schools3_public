from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.decision_tree_hyperparameters_config as config

class DecisionTreeHyperparameters(Hyperparameters):
    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=None, seed=None):
        super(DecisionTreeHyperparameters, self).__init__()
        self.max_depth = config.get_max_depth_hp(max_depth)
        self.min_samples_leaf = config.get_min_samples_leaf_hp(min_samples_leaf)
        self.min_samples_split = config.get_min_samples_split_hp(min_samples_split)
        self.seed = config.get_seed_hp(seed)
