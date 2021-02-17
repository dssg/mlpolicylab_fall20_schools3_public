import argparse
import warnings
import pandas as pd
from schools3.config import main_config as config
from schools3.ml.experiments import *

warnings.filterwarnings('ignore')

def add_args(parser):
    parser.add_argument('--name', default='ignore', help='name of current experiment')
    parser.add_argument('--exp', default='SingleDatasetExperiment', help='name of experiment to run')
    parser.add_argument('--no-good-values', action='store_false')
    parser.add_argument('--no-cache', action='store_false')

# converts the string name of an Experiment to the corresponding experiment's class
def resolve_experiment(exp):
    return globals()[exp]

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args()

    exp_type = resolve_experiment(args.exp)

    if exp_type == HPTuningExperiment or isinstance(exp_type(), ModelsExperiment):
        exp = exp_type(use_cache=args.no_cache)
    else:
        exp = exp_type()

    if isinstance(exp, HPTuningExperiment):
        out = exp.perform(good_values_only=args.no_good_values)
    else:
        out = exp.perform()
    print(out)

if __name__ == "__main__":
    main()
