from schools3.ml.base.hyperparameters import *
import schools3.config.ml.hyperparameters.mlp_hyperparameters_config as config


class MLPHyperparameters(Hyperparameters):
    def __init__(self, **kwargs):
    #hidden_sizes=None, loss=None, optimizer=None, epochs=None, batch_size=None):
        super(MLPHyperparameters, self).__init__()
        self.hidden_sizes = config.get_hidden_sizes_hp(kwargs.get('hidden_sizes', None))
        self.loss = config.get_loss_hp(kwargs.get('loss', None))
        self.optimizer = config.get_optimizer_hp(kwargs.get('optimizer', None))
        self.epochs = config.get_epochs_hp(kwargs.get('epochs', None))
        self.batch_size = config.get_batch_size_hp(kwargs.get('batch_size', None))
        self.reg_const = config.get_reg_const_hp(kwargs.get('reg_const', None))
