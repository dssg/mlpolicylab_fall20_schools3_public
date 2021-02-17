from sklearn.ensemble import RandomForestClassifier
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.ml.hyperparameters.random_forest_hyperparameters import RandomForestHyperparameters
import schools3.config.global_config as global_config

class RandomForestModel(SklearnModel):
    def __init__(self, hps:RandomForestHyperparameters=None):
        if hps is None:
            hps = RandomForestHyperparameters()

        model = RandomForestClassifier(
            n_estimators=hps.n_estimators, 
            max_depth=hps.max_depth,
            random_state=hps.seed,
            max_features=hps.max_features,
            class_weight='balanced',
            n_jobs=global_config.num_threads,
        )

        super(RandomForestModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return RandomForestHyperparameters
