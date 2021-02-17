from sklearn.neighbors import KNeighborsClassifier
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.ml.hyperparameters.k_neighbors_hyperparameters import KNeighborsHyperparameters
import schools3.config.global_config as global_config

class KNeighborsModel(SklearnModel):
    def __init__(self, hps:KNeighborsHyperparameters=None):
        if hps is None:
            hps = KNeighborsHyperparameters()

        model = KNeighborsClassifier(
            n_neighbors=hps.n_neighbors, 
            n_jobs=global_config.num_threads,
        )

        super(KNeighborsModel, self).__init__(
            core_model=model,
            hps=hps
        )

    @classmethod
    def get_hp_type(cls):
        return KNeighborsHyperparameters