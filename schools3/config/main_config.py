import sys
import pandas as pd
import sherpa
from schools3.config import base_config
from schools3.data.features.processors.impute_null_processor import ImputeNullProcessor, ImputeBy
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.features.processors.standardize_processor import StandardizeProcessor
from schools3.ml.models.ensemble_model import EnsembleModel
from schools3.ml.models.logistic_regression_model import LogisticRegressionModel
from schools3.ml.models.random_forest_model import RandomForestModel
from schools3.ml.models.all_ones_model import AllOnesModel
from schools3.ml.models.gradient_boosting_model import GradientBoostingModel
from schools3.ml.models.svm_model import SVMModel
from schools3.ml.models.k_neighbors_model import KNeighborsModel
from schools3.ml.models.all_ones_model import AllOnesModel
from schools3.ml.models.mlp_model import MLPModel
from schools3.data.features import snapshot_features, academic_features, inv_features, oaaogt_features, absence_desc_features, absence_features, middle_school_features, acs_features, inv_type_features, discipline_incident_rate_features
from schools3.data.labels.original_labels import OriginalLabels
from schools3.ml.metrics.performance_metrics import PerformanceMetrics
from schools3.ml.metrics.composite_metrics import CompositeMetrics
from schools3.ml.metrics.fairness_metrics import FairnessMetrics
from schools3.ml.hyperparameters.gradient_boosting_hyperparameters import GradientBoostingHyperparameters
from schools3.ml.hyperparameters.k_neighbors_hyperparameters import KNeighborsHyperparameters
from schools3.ml.hyperparameters.random_forest_hyperparameters import RandomForestHyperparameters
from schools3.ml.baselines.gpa_baseline import GPABaseline
from schools3.ml.baselines.discipline_baseline import DisciplineBaseline
from schools3.ml.baselines.absenteeism_baseline import AbsenteeismBaseline
from schools3.config.data.datasets import datasets_generator_config


config = base_config.Config()

# values for single dataset experiments
def check_single_dataset(grade, train_years, test_years):
    assert min(test_years) >= max(train_years) + datasets_generator_config.label_windows[grade]

config.single_grade = 10
config.train_years = [y for y in range(2006, 2012)]
config.test_years = [2013]

check_single_dataset(config.single_grade, config.train_years, config.test_years)

# values for multi dataset experiments
config.multi_grades = [10]

config.features = [
    snapshot_features.SnapshotFeatures(),
    academic_features.AcademicFeatures(),
    inv_features.InvFeatures(),
    oaaogt_features.OAAOGTFeatures(),
    absence_features.AbsenceFeatures(),
    absence_desc_features.AbsenceDescFeatures(),
    #discipline_incident_rate_features.DisciplineIncidentRateFeatures()
    # acs_features.ACSFeatures(),
    # middle_school_features.MiddleSchoolFeatures()
]

config.labels = OriginalLabels()

def get_default_feature_processors(train_stats=None):
    train_stats = [None] * 2 if train_stats is None else train_stats
    return CompositeFeatureProcessor([
        ImputeNullProcessor(train_stats=train_stats[0], fill_unspecified=ImputeBy.MEAN, col_flag_set=False),
        StandardizeProcessor(train_stats=train_stats[1])
    ])

config.get_default_feature_processors = get_default_feature_processors

config.model_types = [
    GPABaseline,
    DisciplineBaseline,
    AbsenteeismBaseline,
    RandomForestModel,
    LogisticRegressionModel,
    # MLPModel,
    #GradientBoostingModel,
    #SVMModel,
]

config.models = [x() for x in config.model_types]

config.metrics = CompositeMetrics([
    PerformanceMetrics()
])

config.fairness_attributes = [
    'ethnicity',
    'district'
]

config.overwrite_cache  = False
config.use_cache        = False

# values for HP tuning
def get_sherpa_algorithm():
    algo = sherpa.algorithms.GridSearch()
    return algo

config.get_sherpa_algorithm = get_sherpa_algorithm

sys.modules[__name__] = config
