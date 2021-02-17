from schools3.ml.base.model import Model
import numpy as np

class AllOnesModel(Model):
    def __init__(self):
        super(AllOnesModel, self).__init__(self, None)

    def train(self, dataset, val_dataset):
        pass

    def predict(self, features):
        return np.ones(len(features))

    @classmethod
    def get_hp_type(cls):
        return None
