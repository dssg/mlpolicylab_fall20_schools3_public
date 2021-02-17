from abc import ABC, abstractmethod
import pandas as pd

# base class for a feature processor. These take a Dataframe of features and applies
# some transformations on it. For example, imputation or normalization
class FeatureProcessor(ABC):
    def __init__(self, train_stats=None):
        self.train_stats = train_stats

    # the "main" method of the processor
    @abstractmethod
    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        pass

    # returns any stats that might be needed after processing. 
    # For example, mean/std for a normalizing processor
    def get_stats(self):
        return None
