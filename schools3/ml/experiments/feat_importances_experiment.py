from joblib import Parallel
from joblib import delayed
from schools3.ml.experiments.single_dataset_experiment import SingleDatasetExperiment
from schools3.data.base.cohort import Cohort
from schools3.config import main_config
from schools3.config import global_config
import schools3.config.ml.experiments.feat_importances_experiment_config as config
from schools3.ml.models.tfkeras_model import TFKerasModel
from schools3.ml.models.sklearn_model import SklearnModel
from schools3.data.datasets.dataset import Dataset
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch, check_random_state
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# an experiment that trains models and reports the top features according to permutation importance for a single grade
class FeatureImportancesExperiment(SingleDatasetExperiment):
    def perform(
        self, grade=main_config.single_grade, 
        train_years=main_config.train_years,
        test_years=main_config.test_years,
        *args, **kwargs
    ):
        train_cohort = Cohort(grade, train_years)
        test_cohort = Cohort(grade, test_years)

        df = pd.DataFrame()
        for model in self.models:
            if not (isinstance(model, SklearnModel) or isinstance(model, TFKerasModel)):
                continue

            feature_proc = model.get_feature_processor
            train_data, test_data = \
                self.get_train_test_data(train_cohort, feature_proc, test_cohort)

            model.train(train_data, test_data)

            model_name = model.get_model_name()
            file_name = model_name + '_importances.png'
            save_path = global_config.get_save_path(file_name)

            feature_names, result, sorted_idxs_full = self.get_feature_importances(model, train_data)
            sorted_idxs = sorted_idxs_full[-config.top_n_features:]

            fig, ax1 = plt.subplots()
            ax1.boxplot(
                result.importances[sorted_idxs].T,
                vert=False,
                labels=feature_names[sorted_idxs]
            )
            ax1.set_title('Top Features for {}'.format(model_name))
            ax1.set_xlabel('Importance')

            fig.tight_layout()
            fig.savefig(save_path, facecolor='w')

            cur_df = pd.DataFrame({
                'feature_name': feature_names[sorted_idxs_full[::-1]], 
                'importance_score': result.importances_mean[sorted_idxs_full[::-1]],
            })
            cur_df['model'] = model.get_model_name()
            df = pd.concat([df, cur_df], ignore_index=True)

        return df

    # finds the importance scores for each feature using permutation importance
    # prereq: model must not be a baseline
    def get_feature_importances(self, model, dataset,
        n_repeats=config.n_repeats,
    ):
        X_train, y_train = dataset.get_features_labels()

        # if a sklearn model, just use sklearn's method
        if isinstance(model, SklearnModel):
            result = permutation_importance(model.core_model, X_train, y_train, 
                scoring=config.feat_scorer_sklearn, n_repeats=n_repeats, n_jobs=global_config.num_threads)
        # otherwise, use a modified version of their code
        elif isinstance(model, TFKerasModel):
            scoring = config.feat_scorer
            y_pred = model.predict(X_train)
            baseline_score = scoring(y_train, y_pred)

            all_scores = []
            for col_idx in range(X_train.shape[1]):
                score = self._calculate_permutation_scores(model, X_train, y_train, col_idx, None, n_repeats, scoring)
                all_scores.append(score)

            importances = baseline_score - np.array(all_scores)
            result = Bunch(importances_mean=np.mean(importances, axis=1),
                    importances_std=np.std(importances, axis=1),
                    importances=importances)

        sorted_idxs = result.importances_mean.argsort()
        feature_names = np.array(X_train.columns)

        return feature_names, result, sorted_idxs

    # a modified version of sklearn's permutation importance score. 
    # Does not require sklearn models, and also doesn't do threading
    def _calculate_permutation_scores(self, model, X, y, col_idx, random_state,
        n_repeats, scoring
    ):

        random_state = check_random_state(random_state)
        X_permuted = X.copy()
        scores = np.zeros(n_repeats)
        shuffling_idx = np.arange(X.shape[0])
        for n_round in range(n_repeats):
            random_state.shuffle(shuffling_idx)
            if hasattr(X_permuted, "iloc"):
                col = X_permuted.iloc[shuffling_idx, col_idx]
                col.index = X_permuted.index
                X_permuted.iloc[:, col_idx] = col
            else:
                X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]

            y_pred = model.predict(X_permuted)
            feature_score = scoring(y, y_pred)
            scores[n_round] = feature_score

        return scores