from schools3.ml.base.model import Model
import numpy as np
import pandas as pd


class EnsembleModel(Model):
    def __init__(self, models, ensemble_func=pd.Series.mean):
        super(EnsembleModel, self).__init__(
            core_model=None,
            hyperparameters=None
        )

        self.models = models
        self.ensemble_func = ensemble_func

    def train(self, dataset):
        for m in self.models:
            m.train(dataset)

    def predict(self, features):
        predictions = [m.predict(features) for m in self.models]
        joined_df = pd.DataFrame(np.stack(predictions, axis=1))
        y_hat = joined_df.apply(self.ensemble_func, axis='columns')

        return y_hat.to_numpy()
