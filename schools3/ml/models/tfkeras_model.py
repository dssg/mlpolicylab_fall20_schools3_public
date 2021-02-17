import datetime
import tensorflow as tf
import numpy as np
from schools3.ml.base.model import Model
from schools3.config import global_config
from schools3.config.ml.models import tfkeras_model_config

# a wrapper for models built using the Tensorflow-Keras library
class TFKerasModel(Model):
    def __init__(self, core_model, hps, late_compile=False):
        self.compiled = False
        super(TFKerasModel, self).__init__(core_model, hps)

        if late_compile:
            assert core_model is None
        else:
            self.compile_model()

        self.cacheable = False

    def compile_model(self, x=None):
        self.core_model = self.build_model(x)
        self.core_model.compile(
                loss=self.hps.loss,
                optimizer=self.hps.optimizer,
            )
        self.compiled = True

    def build_model(self, x):
        return self.core_model

    def get_model_name(self):
        return self.__class__.__name__    

    def get_xy(self, dataset):
        x, y = dataset.get_features_labels()
        x = np.asarray(x).astype(np.float32)
        y = np.asarray(y).astype(np.int32)

        return x, y

    def train(self, train_dataset, val_dataset):
        assert val_dataset is not None, 'val_dataset is needed to perform early stopping'
        X_train, y_train = self.get_xy(train_dataset)
        X_val, y_val     = self.get_xy(val_dataset)

        if not self.compiled:
            self.compile_model(X_train)

        log_dir = global_config.get_save_path(tfkeras_model_config.tensorboard_log_dir, use_user_time=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=tfkeras_model_config.patience, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        self.core_model.fit(
            x=X_train,
            y=y_train,
            epochs=self.hps.epochs,
            batch_size=self.hps.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )

    def predict(self, features):
        features = np.asarray(features).astype(np.float32)
        return self.core_model.predict(features)
