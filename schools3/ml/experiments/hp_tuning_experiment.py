import json
import atexit
import sherpa
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn
import matplotlib.pyplot as plt
import plotly.express as px
from schools3.ml.base.experiment import Experiment
from schools3.ml.experiments.single_dataset_experiment import SingleDatasetExperiment
from schools3.ml.experiments.multi_dataset_experiment import MultiDatasetExperiment
from schools3.config import main_config
from schools3.ml.models import *
from schools3.ml.hyperparameters import *
from schools3.ml.base.hyperparameters import HPTuneLevel
from schools3.config import global_config
import schools3.config.ml.experiments.hp_tuning_experiment_config as config

# our model-ranking function
def training_data_weighted_rank(row):
    idxs = np.argsort(row['test_cohort'])
    ws = np.array(row['train_rows']) / row['train_rows'][idxs[-1]]
    normalized_ws = ws / ws.sum()
    return np.dot(np.array(row['test precision using top 5.0%']), normalized_ws)

# an experiment that does hyperparameter grid search using the sherpa library
class HPTuningExperiment(Experiment):
    def __init__(
        self, name='ignore', model_types=main_config.model_types,
        get_algorithm=main_config.get_sherpa_algorithm,
        criteria='test precision using top 5.0%',
        features_list=main_config.features,
        labels=main_config.labels,
        metrics=main_config.metrics,
        use_multi_dataset=True,
        lower_is_better=False,
        use_cache=main_config.use_cache
    ):
        super(HPTuningExperiment, self).__init__(name, features_list, labels)
        self.model_types = model_types
        self.get_algorithm = get_algorithm
        self.criteria = criteria
        self.metrics = metrics
        self.use_multi_dataset = use_multi_dataset
        self.out_csv = global_config.get_save_path(config.out_csv, use_user_time=True)
        self.out_img = global_config.get_save_path(config.out_img, use_user_time=True)
        self.metrics_df = pd.DataFrame()
        self.lower_is_better = lower_is_better
        self.use_cache = use_cache

    # iterates through the hyperparameter grid and evaluates each one.
    # Returns all results as well as the best model + hyperparameters
    def perform(self,
        grade=main_config.single_grade,
        tune_level=HPTuneLevel.HIGH,
        good_values_only=False
    ):
        atexit.register(self.save_df_plot)
        best_hp_dict = {}
        for model_type in tqdm(self.model_types):
            hp_type = model_type.get_hp_type()

            if hp_type is not None:
                hps = hp_type()
                study = sherpa.Study(
                    parameters=hps.get_sherpa_parameters(
                        good_values_only=good_values_only,
                        tune_level=tune_level
                    ),
                    algorithm=self.get_algorithm(),
                    lower_is_better=self.lower_is_better
                )

                for trial in tqdm(study):
                    hyperparameters = hp_type()
                    hyperparameters.load_from_dict(trial.parameters)
                    model = model_type(hyperparameters)
                    cur_metrics = self.get_metrics(grade, model)
                    val = cur_metrics[self.criteria].mean()
                    self.metrics_df = pd.concat([self.metrics_df, cur_metrics], ignore_index=True)

                    study.add_observation(trial=trial, iteration=0, objective=val)
                    study.finalize(trial)

                best_hp_dict[model.get_model_name()] = study.get_best_result()
            else:
                model = model_type()
                cur_metrics = self.get_metrics(grade, model)
                self.metrics_df = pd.concat([self.metrics_df, cur_metrics], ignore_index=True)

        self.save_df_plot()
        atexit.unregister(self.save_df_plot)

        return self.metrics_df, best_hp_dict

    # save the evaluation results to a CSV and plots the performance
    def save_df_plot(self):
        self.metrics_df.to_csv(self.out_csv, index=False)
        self.save_multi_metrics_plot(self.metrics_df)

    # converts stringified hyperparameters into a list of underscore-separated values
    def convert_hps_col(self, x):
        if x is None:
            return ''
        else:
            return '_'.join([
                str(round(v, 1)) if isinstance(v, float) else str(v)
                for k, v in sorted(json.loads(x).items())
            ])

    # traind and evaluate a specific model, then return metrics
    def get_metrics(self, grade, model):
        if self.use_multi_dataset:
            experiment = MultiDatasetExperiment(name='ignore', models=[model], metrics=self.metrics, use_cache=self.use_cache)
            return experiment.perform(grades=[grade])
        else:
            experiment = SingleDatasetExperiment(name='ignore', models=[model], metrics=self.metrics, use_cache=self.use_cache)
            return experiment.perform()

    # plots the performance across years for all the model results in a Dataframe
    def save_multi_metrics_plot(self, df):
        seaborn.set_style('darkgrid')

        plot_df = df[['model', 'hps', 'test_cohort', self.criteria]]

        plot_df['group_by'] = plot_df['model'] + '_' + \
                                plot_df['hps'].apply(self.convert_hps_col)

        fig = px.line(
            plot_df,
            x='test_cohort',
            y=self.criteria,
            color='group_by'
        )
        fig.write_html(self.out_img)

    # applies our ranking method to find the "best" model given its performance on multiple years
    def rank_models(self, model_df, ranking_method=training_data_weighted_rank):
        gdf = model_df.groupby(['model', 'hps']).agg(list)

        return gdf.apply(ranking_method, axis=1).sort_values(ascending=False)
