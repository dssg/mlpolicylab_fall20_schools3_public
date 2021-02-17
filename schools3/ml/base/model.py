import nujson
import pandas as pd
from schools3.data.datasets.dataset import Dataset
from schools3.ml.base.hyperparameters import Hyperparameters
from schools3.config.ml.metrics import performance_metrics_config

# base model class, which typically wraps around an sklearn or tensorflow model and provides
# a common interface for the rest of the code. 
class Model():
    def __init__(self, core_model, hps: Hyperparameters):
        self.core_model = core_model
        self.hps = hps
        self.feature_train_stats = None
        self.cacheable = True

    def get_model_name(self):
        return self.core_model.__class__.__name__

    # convert the model's hyperparameters to JSON format
    def jsonify_hps(self):
        return nujson.dumps(self.hps.get_val_dict()) if self.hps else None

    # trains the model and returns the result of testing it
    def train_test(self, train_dataset: Dataset, test_dataset: Dataset, use_test_as_val=True):
        self.train(train_dataset, test_dataset if use_test_as_val else None)
        return self.test(test_dataset)

    def train(self, train_dataset: Dataset, val_dataset: Dataset=None):
        raise NotImplementedError

    # returns a list of score predictions given a Dataframe of features
    def predict(self, features: pd.DataFrame):
        raise NotImplementedError

    # a place for model-specific FeatureProcessors
    def get_feature_processor(self, train_stats=None):
        import schools3.config.main_config as main_config
        return main_config.get_default_feature_processors(train_stats)

    # abstract method to link each model class with its corresponding hyperparameter class
    @classmethod
    def get_hp_type(cls):
        raise NotImplementedError

    # predicts scores and returns in a Dataframe with student lookup and labels
    def test(self, dataset: Dataset):
        X_test, y_test = dataset.get_features_labels()

        y_hat = self.predict(X_test)

        results_df = y_test
        results_df['score'] = y_hat

        return results_df

    # predicts labels by treating the top k% scoring rows as positive class
    def predict_labels(self, dataset: Dataset, return_full_df=False):
        features, labels = dataset.get_features_labels()
        pred_labels = self.test(dataset).score.sort_values(ascending=False)
        num_positives = int(len(pred_labels) * performance_metrics_config.best_precision_percent)
        pred_labels[:num_positives] = 1
        pred_labels[num_positives:] = 0
        pred_labels.name = 'pred_label'

        if return_full_df:
            return pd.concat({
                'features': features,
                'pred_labels': pred_labels,
                'labels': labels
            }, axis=1)
        else:
            return pred_labels
