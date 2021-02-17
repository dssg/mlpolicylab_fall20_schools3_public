import numpy as np
import pandas as pd
from schools3.ml.experiments.models_experiment import ModelsExperiment
from schools3.data.base.cohort import Cohort
from schools3.config import main_config
from schools3.config import global_config
from schools3.data.datasets.dataset import Dataset
from schools3.ml.experiments.feat_importances_experiment import FeatureImportancesExperiment
from schools3.ml.experiments.single_dataset_experiment import SingleDatasetExperiment
from schools3.ml.models.tfkeras_model import TFKerasModel
from schools3.ml.models.sklearn_model import SklearnModel
import schools3.config.ml.experiments.feat_pruning_experiment_config as config
from schools3.config.data.datasets import dataset_config

# an experiment that trains models with subsets of the features according to their permutation importance rank
# like SingleDatasetExperiment, this works on a specific grade
class FeaturePruningExperiment(ModelsExperiment):
    def __init__(
        self, name='ignore',
        features_list=main_config.features,
        labels=main_config.labels,
        models=main_config.models,
        metrics=main_config.metrics,
        use_cache=main_config.use_cache
    ):
        super(FeaturePruningExperiment, self).__init__(
            name, features_list, labels, models, metrics, use_cache=use_cache
        )

    def perform(
        self, grade=main_config.single_grade,
        train_years=main_config.train_years,
        test_years=main_config.test_years,
        compute_train_metrics=False, **kwargs
    ):
        train_cohort = Cohort(grade, train_years)

        df = pd.DataFrame()
        for model in self.models:
            if not (isinstance(model, SklearnModel) or isinstance(model, TFKerasModel)):
                continue

            train_data = Dataset(train_cohort, self.features_list, model.get_feature_processor(), self.labels)
            model.train(train_data)

            feats_exp = FeatureImportancesExperiment('ignore', self.features_list, self.labels, [model], self.metrics)
            
            feature_names, _, sorted_idxs = feats_exp.get_feature_importances(model, train_data)
            feats = np.flip(feature_names[sorted_idxs])

            for i in config.num_feats:
                dataset_config.feat_whitelist.clear()
                for feat in feats[:i]:
                    dataset_config.feat_whitelist.append(feat)

                exp = SingleDatasetExperiment('ignore', self.features_list, self.labels, [model], self.metrics)
                cur_df = exp.perform(grade, train_years, test_years, compute_train_metrics=compute_train_metrics, **kwargs)
                cur_df['num_feats'] = i

                df = pd.concat([df, cur_df], ignore_index=True)

        return df
