from sklearn.ensemble import GradientBoostingClassifier
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.ml.hyperparameters.gradient_boosting_hyperparameters import GradientBoostingHyperparameters

class GradientBoostingModel(SklearnModel):
    def __init__(self, hps:GradientBoostingHyperparameters=None):
        if hps is None:
            hps = GradientBoostingHyperparameters()

        model = GradientBoostingClassifier(
            learning_rate=hps.learning_rate,
            n_estimators=hps.n_estimators,
            max_depth=hps.max_depth,
        )

        super(GradientBoostingModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return GradientBoostingHyperparameters