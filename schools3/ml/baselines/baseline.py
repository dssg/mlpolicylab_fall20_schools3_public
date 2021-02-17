from schools3.ml.base.model import Model

# base class for baselines, which can make predictions but does not train an ML model
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__(core_model=None, hps=None)
        self.cacheable = False

    def get_model_name(self):
        return self.__class__.__name__

    def train(self, dataset, val_data=None):
        pass # no training needed for baselines

    def predict(self, features):
        pass

    @classmethod
    def get_hp_type(cls):
        return None

    def test(self, dataset):
        _, labels = dataset.get_features_labels() # ignore features
        return self.get_scores(dataset.cohort, labels)

    def get_scores(self, cohort, labels):
        raise NotImplementedError
