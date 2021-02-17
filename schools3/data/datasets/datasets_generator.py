from typing import List
from schools3.data.datasets.dataset import Dataset
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.feature_processor import FeatureProcessor
from schools3.data.base.cohort import Cohort
from schools3.data.labels.labels_table import LabelsTable
from schools3.config.data.datasets import datasets_generator_config as config


class DatasetsGenerator():
    def __init__(self, grade):
        self.grade = grade
        self.years = sorted(config.years_per_grade[self.grade])

    def get_all_train_test_pairs(self, include_all_train_hist=True):
        for train_year in self.years:
            test_year = train_year + config.label_windows[self.grade]
            if test_year not in self.years:
                break

            if include_all_train_hist:
                train_years = [y for y in self.years if y <= train_year]
            else:
                train_years = [train_year]

            train_cohort = Cohort(self.grade, train_years)
            test_cohort = Cohort(self.grade, [test_year])

            yield train_cohort, test_cohort
