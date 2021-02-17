import os
import pickle
import uuid
import pandas as pd
from schools3.ml.base.experiment import Experiment
from schools3.data.features.processors.composite_feature_processor import CompositeFeatureProcessor
from schools3.data.datasets.dataset import Dataset
from schools3.config import main_config
from schools3.config import global_config
from schools3.config.ml.experiments import models_experiment_config
from schools3.ml.metrics.fairness_metrics import FairnessMetrics
from schools3.config.data.features.processors import categorical_feature_processor_config as cat_config
from schools3.data.features.processors.categorical_feature_processor import CategoricalFeatureProcessor

# base abstract class for experiments that train models and report metrics on their predictions
class ModelsExperiment(Experiment):
    def __init__(
        self, name='ignore',
        features_list=main_config.features,
        labels=main_config.labels,
        models=main_config.models,
        metrics=main_config.metrics,
        categorical_fairness_attributes=main_config.fairness_attributes,
        use_cache=main_config.use_cache
    ):
        super(ModelsExperiment, self).__init__(name, features_list, labels)
        self.models = models
        self.metrics = metrics
        self.categorical_fairness_attributes = categorical_fairness_attributes
        self.get_model_csv_cache = models_experiment_config.get_model_csv_cache
        self.use_cache = use_cache

    # evaluates each model and returns all metrics in a Dataframe
    def get_train_test_metrics(self, train_cohort, test_cohort, compute_train_metrics=True, use_test_for_val=True):
        df = pd.DataFrame()
        for model in self.models:
            feature_proc = model.get_feature_processor
            train_data, test_data = \
                self.get_train_test_data(train_cohort, feature_proc, test_cohort)
            if test_data.get_dataset().shape[0] < models_experiment_config.min_test_rows:
                continue

            cached_model = self.maybe_get_cached_model(model, train_data, test_data)
            if cached_model is None:
                model.train(train_data, test_data if use_test_for_val else None)
                self.maybe_cache_model(model, train_data, test_data)
            else:
                model = cached_model

            train_metrics = pd.DataFrame()
            if compute_train_metrics:
                train_scores = model.test(train_data)
                train_metrics = self.metrics.compute(train_scores)

            test_scores = model.test(test_data)
            test_metrics = self.metrics.compute(test_scores)
            test_metrics = pd.concat([
                            test_metrics,
                            self.get_fairness_metrics(model, test_data)
                        ], axis=1)

            cur_df = self.construct_metrics_row(model, train_data, test_data, train_metrics, test_metrics)

            df = pd.concat([df, cur_df], ignore_index=True)

        return df

    # save a given model and update index of cached models
    def maybe_cache_model(self, model, train_data, test_data):
        if not (model.cacheable and self.use_cache):
            return

        key_cols = list(self.get_id_cols())
        cache_df = self.get_cache_df(model)

        id_row = self.construct_id_row(model, train_data, test_data)
        key = tuple(id_row.iloc[0])

        h = str(uuid.uuid4())
        model_file = models_experiment_config.get_hash_pkl_file(model.get_model_name(), h)
        h_fname = global_config.get_save_path(model_file)
        with open(h_fname, 'wb') as f:
            pickle.dump(model, f)

        id_row['hash'] = h
        id_row = id_row.set_index(key_cols)

        if key in cache_df.index:
            cache_df.loc[key] = id_row.iloc[0]
        else:
            cache_df = cache_df.append(id_row)

        cache_csv_path = global_config.get_save_path(self.get_model_csv_cache(model.get_model_name()))
        cache_df.to_csv(cache_csv_path)

    # load a given model if it has been cached
    def maybe_get_cached_model(self, model, train_data, test_data):
        if main_config.overwrite_cache or (not self.use_cache):
            return None

        id_row = self.construct_id_row(model, train_data, test_data)
        key = tuple(id_row.iloc[0])
        cache_df = self.get_cache_df(model)

        if key not in cache_df.index:
            return None

        h = cache_df.loc[key].hash
        model_file = models_experiment_config.get_hash_pkl_file(model.get_model_name(), h)
        h_fname = global_config.get_save_path(model_file)

        with open(h_fname, 'rb') as f:
            model = pickle.load(f)

        return model

    # read the CSV file that lists all cached models, and return this list as a Dataframe
    def get_cache_df(self, model):
        cache_csv_path = global_config.get_save_path(self.get_model_csv_cache(model.get_model_name()))
        key_cols = list(self.get_id_cols())
        return pd.read_csv(cache_csv_path).set_index(key_cols) if os.path.exists(cache_csv_path) else pd.DataFrame()

    # creates Dataset objects for the given cohorts
    def get_train_test_data(self, train_cohort, get_processors, test_cohort):
        train_proc = get_processors()
        train_data = Dataset(train_cohort, self.features_list, train_proc, self.labels)
        train_stats = train_data.get_feature_proc_stats()

        test_proc = get_processors(train_stats)
        test_data = Dataset(test_cohort, self.features_list, test_proc, self.labels)
        return train_data, test_data

    # helper method to construct one row of metrics in the form of a Dataframe
    def construct_metrics_row(
        self, model, train_data, test_data, train_metrics, test_metrics
    ):
        train_metrics = train_metrics.rename(columns={c: f'train {c}' for c in train_metrics.columns})
        test_metrics  = test_metrics.rename(columns={c: f'test {c}' for c in test_metrics.columns})
        id_row = self.construct_id_row(model, train_data, test_data)

        df = pd.concat([id_row, train_metrics, test_metrics], axis=1)

        return df

    # helper method to get each row's identifier values in a Dataframe
    def construct_id_row(self, model, train_data, test_data):
        df = pd.DataFrame()

        model_col, hps_col, train_cohort_col, test_cohort_col, train_rows, test_rows, num_features = self.get_id_cols()

        df[model_col]           = [model.get_model_name()]
        df[hps_col]             = [model.jsonify_hps()]
        df[train_cohort_col]    = [train_data.cohort.get_identifier()]
        df[test_cohort_col]     = [test_data.cohort.get_identifier()]
        df[train_rows]          = [train_data.get_dataset().shape[0]]
        df[test_rows]           = [test_data.get_dataset().shape[0]]
        df[num_features]        = [len(train_data.get_dataset().columns)]

        return df

    # helper method that specifies column names for each row's identifiers
    def get_id_cols(self):
        model_col           = 'model'
        hps_col             = 'hps'
        train_cohort_col    = 'train_cohort'
        test_cohort_col     = 'test_cohort'
        train_rows          = 'train_rows'
        test_rows           = 'test_rows'
        num_features        = 'num_features'

        return model_col, hps_col, train_cohort_col, test_cohort_col, train_rows, test_rows, num_features


    def get_all_fairness_cols(self, dataset):
        cs = dataset.get_dataset().features.columns
        ret_cols = {}
        cat_proc = CategoricalFeatureProcessor()
        for f in self.categorical_fairness_attributes:
            ret_cols[f] = list(cat_proc.get_categorical_feature_names(f, cs))
        return ret_cols

    def get_raw_fairness_metrics(self, model, dataset):
        preds = model.predict_labels(dataset, return_full_df=True)
        metrics = FairnessMetrics()
        fairness_dict = {}
        cols = self.get_all_fairness_cols(dataset)
        for orig_c in cols:
            for c in cols[orig_c]:
                assert c in preds.features.columns, f'fairness attribute {c} is not an input feature'

                grouped_labels = preds[preds.features[c] == 1].groupby(('features', c))\
                    [('pred_labels', 'pred_label'), ('labels', 'label')].agg(list)

                if orig_c not in fairness_dict:
                    fairness_dict[orig_c] = {}

                fairness_dict[orig_c][c] = metrics.compute(
                    metrics.get_score_df(
                        grouped_labels.loc[1][0],
                        grouped_labels.loc[1][1]
                    )
                ).to_dict('list')

        fairness_dict = \
            {
                (cat, v): fairness_dict[cat][v]
                for cat in fairness_dict
                for v in fairness_dict[cat]
            }

        for k1 in fairness_dict:
            for k2 in fairness_dict[k1]:
                assert len(fairness_dict[k1][k2]) == 1
                fairness_dict[k1][k2] = fairness_dict[k1][k2][0]

        return pd.DataFrame.from_dict(fairness_dict, orient='index')

    def get_fairness_metrics(self, model, dataset, ref_cols=models_experiment_config.ref_cols):
        raw_metrics = self.get_raw_fairness_metrics(model, dataset)
        df = pd.DataFrame()
        for cat in raw_metrics.index.get_level_values(0).unique():
            if cat in ref_cols:
                if isinstance(ref_cols[cat], tuple):
                    numer = raw_metrics.loc[cat].loc[ref_cols[cat][0]].median()
                    denom = raw_metrics.loc[cat].loc[ref_cols[cat][1]].median()
                    bias = numer / denom
                    cat_metrics = pd.DataFrame.from_dict({cat + ' median ratio': [bias[0]]})
                else:
                    rel_metrics = raw_metrics.loc[cat] / raw_metrics.loc[(cat, ref_cols[cat])]
                    d = rel_metrics.to_dict()
                    d = {
                        (k1 + ': ' + f'{k2} / {ref_cols[cat]}'):[d[k1][k2]]
                        for k1 in d for k2 in d[k1]
                    }
                    cat_metrics = pd.DataFrame.from_dict(d)
            else:
                std = raw_metrics.loc[cat].std().to_dict()
                cat_metrics = pd.DataFrame.from_dict({(f'std {cat} ' + k):[std[k]] for k in std})

            df = pd.concat([df, cat_metrics], axis=1)

        return df
