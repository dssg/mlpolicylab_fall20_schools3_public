import pandas as pd
from tqdm import tqdm
from schools3.data.datasets.datasets_generator import DatasetsGenerator
from schools3.ml.experiments.models_experiment import ModelsExperiment
from schools3.config import main_config

# an experiment that trains models and reports metrics for multiple grades 
class MultiDatasetExperiment(ModelsExperiment):
    def __init__(
        self, name='ignore', features_list=main_config.features,
        labels=main_config.labels, models=main_config.models,
        metrics=main_config.metrics, use_cache=main_config.use_cache
    ):
        super(MultiDatasetExperiment, self).__init__(
            name, features_list, labels, models, metrics, use_cache=use_cache
        )

    def perform(
        self, grades=main_config.multi_grades, include_all_train_hist=True,
        *args, **kwargs
    ):
        df = pd.DataFrame()
        t_grades = tqdm(grades)
        for grade in t_grades:
            t_grades.set_description(f'grade {grade}:')
            generator = DatasetsGenerator(grade)

            cohorts = \
                tqdm(generator.get_all_train_test_pairs(include_all_train_hist))

            for train_cohort, test_cohort in cohorts:
                cohorts.set_description(train_cohort.get_identifier())
                metrics_df = self.get_train_test_metrics(train_cohort, test_cohort)
                df = pd.concat([df, metrics_df], ignore_index=True)

        return df
