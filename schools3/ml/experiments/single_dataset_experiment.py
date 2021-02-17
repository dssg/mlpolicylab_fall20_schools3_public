from schools3.data.base.cohort import Cohort
from schools3.data.datasets.dataset import Dataset
from schools3.ml.experiments.models_experiment import ModelsExperiment
from schools3.config import main_config

# an experiment that trains models and reports metrics for a single grade
class SingleDatasetExperiment(ModelsExperiment):
    def __init__(
        self, name='ignore',
        features_list=main_config.features,
        labels=main_config.labels,
        models=main_config.models,
        metrics=main_config.metrics,
        use_cache=main_config.use_cache
    ):
        super(SingleDatasetExperiment, self).__init__(
            name, features_list, labels, models, metrics, use_cache=use_cache
        )

    def perform(
        self, grade=main_config.single_grade,
        train_years=main_config.train_years, test_years=main_config.test_years,
        compute_train_metrics=True, **kwargs
    ):
        train_cohort = Cohort(grade, train_years)
        test_cohort = Cohort(grade, test_years)

        return self.get_train_test_metrics(
                train_cohort,
                test_cohort,
                compute_train_metrics
            )
