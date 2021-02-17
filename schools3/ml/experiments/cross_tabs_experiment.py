import pandas as pd
from schools3.data.base.cohort import Cohort
from schools3.data.datasets.dataset import Dataset
from schools3.ml.experiments.models_experiment import ModelsExperiment
from schools3.config import main_config
from schools3.config.ml.experiments import models_experiment_config
from schools3.config.data.features.processors import categorical_feature_processor_config
import schools3.config.ml.experiments.cross_tabs_experiment_config as config

# an experiment that trains models and reports the features with greatest value disparity between the positive and negative class
# like SingleDatasetExperiment, this works on a specific grade
class CrossTabsExperiment(ModelsExperiment):
    def __init__(
        self, name='ignore',
        features_list=main_config.features,
        labels=main_config.labels,
        models=main_config.models,
        metrics=main_config.metrics,
        use_cache=main_config.use_cache
    ):
        super(CrossTabsExperiment, self).__init__(
            name, features_list, labels, models, metrics, use_cache=use_cache
        )

    def perform(
        self, grade=main_config.single_grade,
        train_years=main_config.train_years, 
        test_years=main_config.test_years,
        num_features=config.num_features,
        compute_train_metrics=True, **kwargs
    ):
        train_cohort = Cohort(grade, train_years)
        test_cohort = Cohort(grade, test_years)

        df = pd.DataFrame()
        for model in self.models:
            feature_proc = model.get_feature_processor
            train_data, test_data = \
                self.get_train_test_data(train_cohort, feature_proc, test_cohort)

            if test_data.get_dataset().shape[0] < models_experiment_config.min_test_rows:
                continue

            model.train(train_data, test_data)

            feature_preds = model.predict_labels(test_data, True)

            pred0 = feature_preds[feature_preds['pred_labels']['pred_label'] == 0]
            pred1 = feature_preds[feature_preds['pred_labels']['pred_label'] == 1]

            rows = []
            for c in feature_preds['features'].columns:
                col0 = pred0['features'][c]
                col1 = pred1['features'][c]

                if c.startswith(categorical_feature_processor_config.prefix):
                    col0 = col0.astype('int32')
                    col1 = col1.astype('int32')

                pred0_mean = col0.mean()
                pred1_mean = col1.mean()
                rows.append((abs(pred0_mean - pred1_mean), pred1_mean, pred0_mean, c))

            rows.sort(reverse=True)

            cur_df_vals = {k: [] for k in ['model', 'feat_name', 'mean_diff', 'top_mean', 'bottom_mean']}
            for row in rows[:num_features]:
                cur_df_vals['model'].append(model.get_model_name())
                cur_df_vals['feat_name'].append(row[3])
                cur_df_vals['mean_diff'].append(row[0])
                cur_df_vals['top_mean'].append(row[1])
                cur_df_vals['bottom_mean'].append(row[2])

            cur_df = pd.DataFrame({k: cur_df_vals[k] for k in cur_df_vals})
            df = pd.concat([df, cur_df], ignore_index=True)

        return df
