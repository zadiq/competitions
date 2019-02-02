import numpy as np
import os
import tensorflow as tf
import random
import pickle
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import adam
from loans.utils import TRAIN_MODE, EVAL_MODE, TEST_MODE
from loans.data import LoanData
from sklearn.utils.class_weight import compute_class_weight


random.seed(21)
tf.set_random_seed(21)


class TrainConfig:
    model = 0
    epochs = 1000
    model_dir = '/media/zadiq/ZHD/datasets/home_credit/models/'
    model_folder = None
    """ 
    model_folder:
        these are folders generated in model_dir whenever a new training starts. 
        The contain logs and models at check points
        """
    use_class_weights = True
    optimizer_params = {
        'lr': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8,
        'decay': 0.0,
        'amsgrad': False,
    }
    optimizer = adam
    batch_size = 100
    merged_model = {
        'layers': [1000, 1500, 2000, 1000, 1750],
        'params': {},
        'dropouts': None,  # or [None, .1, None ..., ]
    }
    workers = 1


# class BuildModel:
#
#     def __init__(self, mode, app_param):
#         options = [utils.TRAIN_MODE, utils.TEST_MODE]
#         assert mode in options, 'mode has to be of these {}. {} provided'.format(options, mode)
#
#         self.mode = mode
#         self.app_param = app_param
#
#         with open(utils.get_file('re_meta')) as meta_fp:
#             self.app_meta_data = json.load(meta_fp)
#
#         if mode == utils.TRAIN_MODE:
#             self.app_meta_data['train_gen'] = self.app_param.get('train_gen')
#             self.app_meta_data['eval_gen'] = self.app_param.get('eval_gen')
#         else:
#             self.app_meta_data['test_gen'] = self.app_param.get('test_gen')
#
#         # change class weights keys to int
#         class_weights = {
#             0: self.app_meta_data['class_weights'].pop('0'),
#             1: self.app_meta_data['class_weights'].pop('1')
#         }
#
#         if TrainConfig.bias_class_weight:
#             class_weights[0] *= TrainConfig.bias_class_value[0]
#             class_weights[1] *= TrainConfig.bias_class_value[1]
#
#         self.app_meta_data['class_weights'] = class_weights
#
#     @staticmethod
#     def auc_roc(y_true, y_pred):
#         # any tensorflow metric
#         value, update_op = tf.metrics.auc(y_true, y_pred)
#
#         # find all variables created for this metric
#         metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
#
#         # Add metric variables to GLOBAL_VARIABLES collection.
#         # They will be initialized for new session.
#         for v in metric_vars:
#             tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
#
#         # force to update metric values
#         with tf.control_dependencies([update_op]):
#             value = tf.identity(value)
#             return value
#
#     def build_simple_app_layers(self):
#         """a simple model for application"""
#         inputs = Input(shape=(self.app_meta_data['col_size'],))
#         layers_params = {
#             'activation': 'relu',
#             'kernel_initializer': 'glorot_normal',
#             'bias_initializer': 'he_normal',
#             'kernel_regularizer': l1_l2(),
#         }
#         x = Dense(100, **layers_params)(inputs)
#         x = Dense(100, **layers_params)(x)
#         x = Dropout(.2)(x)
#         x = Dense(70, **layers_params)(x)
#         x = Dropout(.3)(x)
#         x = Dense(50, **layers_params)(x)
#         x = Dropout(.3)(x)
#         x = Dense(300, **layers_params)(x)
#         x = Dropout(.5)(x)
#         x = Dense(300, **layers_params)(x)
#         x = Dropout(.2)(x)
#         x = Dense(100, **layers_params)(x)
#         x = Dropout(.1)(x)
#         x = Dense(100, **layers_params)(x)
#         layers = Dense(1, activation='sigmoid')(x)
#         return inputs, layers
#
#     def fit(self):
#         inputs, layers = self.build_simple_app_layers()
#         model = Model(inputs=inputs, outputs=layers)
#         optimizer = TrainConfig.optimizer or adam(**TrainConfig.optimizer_params)
#         model.compile(optimizer=optimizer, loss='binary_crossentropy',
#                       metrics=['accuracy', self.auc_roc]
#                       )
#
#         # calculate steps per epoch
#         train_size = self.app_meta_data['train_size']
#         eval_size = self.app_meta_data['eval_size']
#         batch_size = TrainConfig.batch_size
#         train_steps = (train_size // batch_size) + (train_size % batch_size > 0)
#         val_steps = (eval_size // batch_size) + (eval_size % batch_size > 0)
#
#         model.fit_generator(
#             self.app_meta_data['train_gen'],
#             steps_per_epoch=train_steps,
#             validation_data=self.app_meta_data['eval_gen'],
#             validation_steps=val_steps,
#             epochs=TrainConfig.epochs,
#             class_weight=self.app_meta_data['class_weights']
#         )
#
#         return model


def auc_roc(y_true, y_pred):
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


class LoanModel:

    def __init__(self, mode, train_config=TrainConfig(), ckpt_model=None, from_ckpt=False):
        options = [TRAIN_MODE, TEST_MODE]
        assert mode in options, 'mode has to be of these {}. {} provided'.format(options, mode)

        self.mode = mode
        if type(train_config) == str:
            print('Loading pickled train_config from {}'.format(train_config))
            self.tc = self.load_pickled_tc(train_config)
        else:
            self.tc = train_config  # train config
        self.model_type = None

        if mode == TRAIN_MODE:
            self.data = self.train_data = LoanData(self.mode, self.tc.batch_size)
            self.train_gen = self.train_data.data_gen()
            self.val_data = LoanData(EVAL_MODE, self.tc.batch_size)
            self.val_gen = self.train_data.data_gen()
            self.class_weights = None
            if self.tc.use_class_weights:
                ones = np.ones(self.train_data.meta['model_train_class_one_size'])
                zeros = np.zeros(self.train_data.meta['model_train_class_zero_size'])
                classes = [0., 1.]
                weights = compute_class_weight('balanced', classes, np.r_[ones, zeros])
                self.class_weights = dict(zip(classes, weights))
            if from_ckpt:
                self.k_model = load_model(
                    os.path.join(self.tc.model_folder, ckpt_model),
                    custom_objects={'auc_roc': auc_roc}
                )
            else:
                self.model_inputs = self.gen_model_inputs()
                self.k_model = self.build_model(self.tc.model)
        else:
            self.data = self.test_data = LoanData(self.mode, self.tc.batch_size)
            self.test_gen = self.test_data.data_gen()
            self.k_model = load_model(
                os.path.join(self.tc.model_folder, ckpt_model),
                custom_objects={'auc_roc': auc_roc}
            )

    def gen_model_inputs(self):
        model_inputs = {}
        for i in self.data.inputs:
            model_inputs[i] = Input(shape=self.data.input_spec[i], name=i)

        return model_inputs

    def build_model(self, m):
        models = {
            0: self.model_0
        }

        return models[m]()

    @staticmethod
    def build_hidden_layers(x, layers, params, dropouts=None):
        dropouts = dropouts or [None for _ in range(len(layers))]
        assert len(dropouts) == len(layers), ("Size of hidden layers doesn't fit dropouts size. Use None "
                                              "on layers where dropouts shouldn't be applied to.")

        for l, d in zip(layers, dropouts):
            x = Dense(l, **params)(x)
            if d is not None:
                x = Dropout(d)(x)
        return x

    def get_model_folder(self, model_type=None, time=None):
        time_format = time or datetime.now().strftime('%m-%d_%H-%M-%S')
        model_type = model_type or self.model_type
        model_folder = 'model_{type}_{time_format}'.format(type=model_type, time_format=time_format)
        return os.path.join(self.tc.model_dir, model_folder)

    def get_callbacks(self):
        callbacks = []
        os.makedirs(self.tc.model_dir, exist_ok=True)

        self.tc.model_folder = model_folder = self.get_model_folder()
        os.makedirs(model_folder, exist_ok=True)

        log_dir = os.path.join(model_folder, 'logs')
        best_models_dir = os.path.join(model_folder, 'best')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_models_dir, exist_ok=True)
        save_format = 'weights.{epoch:02d}-{val_loss:.2f}-{auc_roc:.2f}.hdf5'
        _j = os.path.join

        callbacks.append(ModelCheckpoint(_j(model_folder, save_format)))
        callbacks.append(ModelCheckpoint(_j(best_models_dir, save_format), save_best_only=True))
        callbacks.append(TensorBoard(log_dir))

        return callbacks

    def calc_steps(self, size):
        bs = self.tc.batch_size
        return (size // bs) + (size % bs > 0)

    def save_tc(self):
        """save a pickled format of train config"""
        with open(os.path.join(self.tc.model_folder, 'train_config.pk'), 'w') as tcp:
            pickle.dump(self.tc, tcp)

    @staticmethod
    def load_pickled_tc(path):
        with open(path) as tcp:
            return pickle.load(tcp)

    def model_0(self):
        self.model_type = 0
        hidden_layers = self.build_hidden_layers(
            x=self.model_inputs['MERGED'],
            **self.tc.merged_model
        )
        outputs = Dense(1, activation='sigmoid')(hidden_layers)

        model = Model(inputs=[self.model_inputs[x] for x in self.data.inputs], outputs=outputs)
        optimizer = self.tc.optimizer(**self.tc.optimizer_params)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy', auc_roc])

        return model

    def fit(self):
        try:
            self.k_model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.calc_steps(self.data.meta['model_train_size']),
                epochs=self.tc.epochs,
                validation_data=self.val_gen,
                validation_steps=self.calc_steps(self.data.meta['model_eval_size']),
                class_weight=self.class_weights,
                workers=self.tc.workers,
                callbacks=self.get_callbacks()
            )
        except KeyboardInterrupt:
            print('Gracefully exiting training')
            self.save_tc()
            self.data.data_fp.close()
            if self.mode == TRAIN_MODE:
                self.val_data.data_fp.close()
