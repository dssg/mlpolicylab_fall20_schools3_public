import sys
from schools3.config import base_config
from sklearn.metrics import roc_curve, auc, make_scorer
import numpy as np

config = base_config.Config()

def permutation_scorer(y_true, y_pred):
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  return auc(fpr, tpr)

config.feat_scorer = permutation_scorer
config.feat_scorer_sklearn = make_scorer(permutation_scorer)
config.top_n_features = 10
config.n_repeats = 1

sys.modules[__name__] = config
