import os
import importlib
from schools3.ml.experiments.single_dataset_experiment import *
from schools3.ml.experiments.multi_dataset_experiment import *
from schools3.ml.experiments.hp_tuning_experiment import *
from schools3.ml.experiments.feat_pruning_experiment import *
from schools3.ml.experiments.models_experiment import *
from schools3.ml.experiments.cross_tabs_experiment import *
from schools3.ml.experiments.local_importances_experiment import *
from schools3.ml.experiments.feat_importances_experiment import *

# def load_everything_from(file):
#     # taken from https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
#     # get a handle on the module
#     mdl = importlib.import_module(file.)

#     # is there an __all__?  if so respect it
#     if "__all__" in mdl.__dict__:
#         names = mdl.__dict__["__all__"]
#     else:
#         # otherwise we import all names that don't begin with _
#         names = [x for x in mdl.__dict__ if not x.startswith("_")]

#     # now drag them in
#     globals().update({k: getattr(mdl, k) for k in names})

# dir_path = os.path.dirname(os.path.realpath(__file__))
# from f in os.listdir(dir_path):
#     load_everything_from(f)
