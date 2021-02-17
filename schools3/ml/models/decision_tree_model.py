from sklearn.tree import DecisionTreeClassifier
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.ml.hyperparameters.decision_tree_hyperparameters import DecisionTreeHyperparameters
import schools3.config.global_config as global_config

class DecisionTreeModel(SklearnModel):
    def __init__(self, hps:DecisionTreeHyperparameters=None):
        if hps is None:
            hps = DecisionTreeHyperparameters()

        model = DecisionTreeClassifier(
            max_depth=hps.max_depth,
            min_samples_split=hps.min_samples_split,
            min_samples_leaf=hps.min_samples_leaf,
            random_state=hps.seed,
            class_weight='balanced',
        )

        super(DecisionTreeModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return DecisionTreeHyperparameters
