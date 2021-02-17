from schools3.ml.experiments.single_dataset_experiment import SingleDatasetExperiment
from schools3.data.base.cohort import Cohort
from schools3.config import main_config
from schools3.config import global_config
import schools3.config.ml.experiments.local_importances_experiment_config as config
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.data.datasets.dataset import Dataset
from schools3.config.ml.metrics import performance_metrics_config

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random

# performs LIME to find locally important features. Currently selects random students to analyze
class LocalImportancesExperiment(SingleDatasetExperiment):
    def perform(
        self, grade=main_config.single_grade, train_years=main_config.train_years, test_years=main_config.test_years,
        *args, **kwargs
    ):
        train_cohort    = Cohort(grade, train_years)
        test_cohort     = Cohort(grade, test_years)

        for model in self.models:
            if isinstance(model, SklearnModel):
                ### get data
                train_data = Dataset(train_cohort, self.features_list, model.get_feature_processor(), self.labels)
                test_data = Dataset(test_cohort, self.features_list, model.get_feature_processor(), self.labels)
                X_test, y_test = test_data.get_features_labels()
                X_train, y_train = train_data.get_features_labels()

                ### train a model
                model.train(train_data)
                model_name = model.get_model_name()

                ### get top-k data
                list_data = self.get_top_k_data(model, test_data)

                ### get an explainer
                #classification
                #explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, model.predict_proba,
                #                                                   feature_names=list(X_train.columns),
                #                                                   class_names=config.class_names,
                #                                                   discretize_continuous=True)

                #exp = explainer.explain_instance(X_test_selected.values[0], model.predict_proba,
                #                                 num_features=config.n_features, top_labels=config.top_labels)

                explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                                   feature_names=list(X_train.columns),
                                                                   class_names=config.class_name,
                                                                   verbose=True,
                                                                   mode='regression')

                for X_test_selected, header in list_data:
                    ### randomly pick samples to analyze
                    list_idx_analyze = random.choices(np.arange(len(X_test_selected)), k=config.n_analyze)

                    ### iterate over random index to analyze
                    for idx in list_idx_analyze:
                        student_lookup = X_test_selected.axes[0].get_level_values('student_lookup').values[idx]
                        file_name = model_name + '_local_importances_' + str(student_lookup) + '.html'
                        path_out = global_config.join_path(global_config.debug_dir, model_name, header)
                        file_out = global_config.join_path_file(path_out, file_name)

                        exp = explainer.explain_instance(X_test_selected.values[idx], model.predict,
                                                         num_features=config.n_features)

                        exp.save_to_file(file_out)

        return


    ### return top-k data with high risk scores
    def get_top_k_data(self, model, test_data, k=performance_metrics_config.best_precision_percent):
        X_test, y_test = test_data.get_features_labels()
        y_test_vals = y_test.values.ravel()

        # predict risk scores
        test_scores = model.test(test_data)
        selected_num = int(test_scores.shape[0] * k)
        sorted_ind = np.argsort(-test_scores['score'].values)
        sorted_ind_low = np.argsort(test_scores['score'].values)

        # predict 1 on the top `self.k` percent
        preds = np.zeros(len(test_scores))
        preds[sorted_ind[:selected_num]] = 1

        # check whether predictions are matched with labels
        matched = np.zeros(y_test_vals.shape)
        matched[preds == y_test_vals] = 1
        X_test['Matched'] = matched

        # get data that has high/low risks
        X_test_high = X_test.iloc[sorted_ind[:selected_num]]
        X_test_low = X_test.iloc[sorted_ind_low[:selected_num]]

        # get data with correct/incorrect predictions for data with high/low risks
        X_test_high_incorrect = X_test_high.loc[X_test_high['Matched'] == 0]
        X_test_high_correct = X_test_high.loc[X_test_high['Matched'] == 1]

        X_test_low_incorrect = X_test_low.loc[X_test_low['Matched'] == 0]
        X_test_low_correct = X_test_low.loc[X_test_low['Matched'] == 1]

        list_data = []
        ### iterate to remove appended prediction labels and include data to return if data length > 1
        for data, header in [[X_test_high_incorrect, 'high_incorrect'], [X_test_high_correct, 'high_correct'],
                            [X_test_low_incorrect, 'low_incorrect'], [X_test_low_correct, 'low_correct']]:
            del data['Matched']
            if (data.shape[0] > 0):
                list_data.append([data, header])

        return list_data