from typing import List
from datetime import datetime
from getpass import getuser
from schools3.data.features.features_table import FeaturesTable
from schools3.data.features.processors.feature_processor import FeatureProcessor
from schools3.data.labels.labels_table import LabelsTable

# base experiment class. This is the top level class that is created in main, which decides
# what task you are trying to run
class Experiment:
    def __init__(
        self, name, features_list: List[FeaturesTable], labels: LabelsTable
    ):
        self.name = name
        self.identifier = self.name + '_' + getuser() + datetime.now().strftime('_%b_%d__%H_%M')
        self.features_list = features_list
        self.labels = labels

    # the "main" method for the Experiment class
    def perform(self, *args, **kwargs):
        raise NotImplementedError
