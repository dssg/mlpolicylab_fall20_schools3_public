from schools3.ml.base.model import Model

# a wrapper for models built using the sklearn library
class SklearnModel(Model):
    def __init__(self, core_model, hps):
        super(SklearnModel, self).__init__(core_model, hps)

    def train(self, train_dataset, val_dataset=None):
        X_train, y_train = train_dataset.get_features_labels()

        self.core_model.fit(X_train, y_train)

    def predict(self, features):
        return self.core_model.predict_proba(features)[:, 1]

    def predict_proba(self, features):
        return self.core_model.predict_proba(features)
