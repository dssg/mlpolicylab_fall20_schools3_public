from sklearn.svm import SVC
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.ml.hyperparameters.svm_hyperparameters import SVMHyperparameters

class SVMModel(SklearnModel):
    def __init__(self, hps:SVMHyperparameters=None):
        if hps is None:
            hps = SVMHyperparameters()

        model = SVC(
            C=hps.inv_reg_strength, 
            kernel=hps.kernel,
            probability=True,
            class_weight='balanced'
        )

        super(SVMModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return SVMHyperparameters