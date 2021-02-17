# inspired by https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/Hyperparameters.py
from enum import IntEnum
import sherpa
import numpy as np
from schools3.config.ml.base import hyperparameters_config as config


class HPTuneLevel(IntEnum):
    HIGH = 3
    MID  = 2
    LOW  = 1
    NA   = 0

# base class for a hyperparameter. This wrapper allows for value-checking and automatic
# interfacing with the sherpa hyperparameter-tuning library
class Hyperparameter(object):
    def __init__(
        self, val, hp_range=[], hp_type='infer',
        log_scale=False, tune_level=HPTuneLevel.HIGH,
        is_categorical=False, categories=None,
        is_none_allowed=False, good_values=[],
        print_func=None
    ):
        assert hp_type == 'infer' or isinstance(hp_type, type), 'invalid value for hp_type'
        assert len(hp_range) == 0 or hp_range[0] <= hp_range[1], 'invalid hp_range'

        self.range              = hp_range
        self.type               = type(val) if hp_type == 'infer' else hp_type
        self.log_scale          = log_scale
        self.tune_level         = tune_level
        self.is_categorical     = is_categorical
        self.categories         = categories
        self.is_none_allowed    = is_none_allowed
        self.good_values        = good_values
        self.print_func         = (lambda x: x) if print_func is None else print_func
        if not self.good_values:
            if self.is_categorical:
                if self.categories is None:
                    self.categories = [val]
                self.good_values = self.categories
            elif self.log_scale:
                self.good_values = list(np.logspace(
                                    self.range[0],
                                    self.range[1],
                                    config.num_good_points,
                                    dtype=self.type
                                ))
            else:
                self.good_values = list(np.linspace(
                                    self.range[0],
                                    self.range[1],
                                    config.num_good_points,
                                    dtype=self.type
                                ))

        self.val = val

    # gets a string representation of this hyperparameter's value
    def get_printable(self):
        if hasattr(self, 'print_func'):
            return self.print_func(self.val)
        else:
            return self.val

    def __setattr__(self, name, val):
        if name == 'val':
            if not self.is_none_allowed:
                assert val is not None, "Hyperparameters value is not allowed to be None"

            if val is not None and len(self.range) > 0:
                assert val >= self.range[0] and val <= self.range[1], f'Hyperparamter value ({val}) out of range ({self.range})'

        object.__setattr__(self, name, val)

    # converts this class into the corresponding class for the sherpa library
    def get_sherpa_parameter(self, name, good_values_only=False):
        if good_values_only:
            return sherpa.Choice(name=name, range=self.good_values)
        elif self.is_categorical:
            return sherpa.Choice(name=name, range=self.categories)
        elif self.type == int:
            return sherpa.Discrete(name=name, range=self.range, scale='log' if self.log_scale else 'linear')
        else:
            return sherpa.Continuous(name=name, range=self.range, scale='log' if self.log_scale else 'linear')

# a class for holding collections of Hyperparameter objects as fields.
# This object is what is actually passed into a Model
class Hyperparameters(object):
    def __init__(self):
        self.__hps = {}

    def __getattr__(self, name):
        if name == '_Hyperparameters__hps':
            return object.__getattribute__(self, name)
        if name in self.__hps:
            return self.__hps[name].val
        raise AttributeError(f'{name} does not exist')

    def __setattr__(self, name, val):
        if name != '_Hyperparameters__hps':
            if name in self.__hps:
                self.__hps[name].val = val
            else:
                assert isinstance(val, Hyperparameter), 'new values must be hyperparameters'
                self.__hps[name] = val
        else:
            object.__setattr__(self, name, val)

    def get_val_dict(self):
        d = {}
        for k in self.__hps:
            d[k] = self.__hps[k].get_printable()

        return d

    def load_from_dict(self, d):
        for k in d:
            assert k in self.__hps, 'key is not a hyperparameter name'
            try:
                self.__hps[k].val = d[k]
            except Exception as e:
                raise ValueError(f'Wrong value for {k}: {str(e)}')

    def get_sherpa_parameters(self, good_values_only=False, tune_level=HPTuneLevel.HIGH):
        params = []
        for k in self.__hps:
            if self.__hps[k].tune_level >= tune_level:
                params.append(self.__hps[k].get_sherpa_parameter(k, good_values_only))
        return params
