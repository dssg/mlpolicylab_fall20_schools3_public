from sklearn.linear_model import LogisticRegression
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.ml.hyperparameters.logistic_regression_hyperparameters import LogisticRegressionHyperparameters
import schools3.config.global_config as global_config

class LogisticRegressionModel(SklearnModel):
    def __init__(self, hps:LogisticRegressionHyperparameters=None):
        if hps is None:
            hps = LogisticRegressionHyperparameters()

        model = LogisticRegression(
            penalty=hps.penalty, 
            max_iter=hps.max_iters, 
            C=hps.inv_reg_strength,
            l1_ratio=hps.l1_ratio, 
            random_state=hps.seed,
            class_weight='balanced', 
            solver='saga',
            n_jobs=global_config.num_threads,
        )

        super(LogisticRegressionModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return LogisticRegressionHyperparameters
