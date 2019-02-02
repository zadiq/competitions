import pandas as pd
import os
import json
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from credit.utils import (
    get_file, TRAIN_MODE,
    TEST_MODE, auc_roc,
    safe_load_json, dump_json
)
from datetime import datetime


OPTIMIZER_MAP = {
    'adam': adam,
}


class TrainConfig:
    def __init__(self):
        self.version = 'v0'
        self.num_folds = 10
        self.stratified = False
        self.seed = 290
        self.use_class_weights = True
        self.board_model = {
            'which': 0,
            'layers': [100, 150, 200, 200, 150, 60],
            'dropouts': [],
            'layer_params': {'activation': 'relu'},
            'use_multi_gpu': True,
            'gpu_counts': 4,
            'optimizer': 'adam',
            'optimizer_params': {},
            'batch_size': 100,
            'epochs': 1000,
            'model_folder': None,
            'test_size': .30,
        }
        self.params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60,  # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
            'device_type': 'cpu',
            'gpu_device_id': 0,
        }

    @classmethod
    def from_json(cls, path):
        obj = cls()
        with open(path) as fp:
            config = json.load(fp)
            obj.__dict__.update(config)
            return obj

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def load_train_config(tc):
    if type(tc) == str:
        return TrainConfig.from_json(tc)
    return tc


class BoardModel:

    def __init__(self, mode=TRAIN_MODE, tc=TrainConfig(), ckpt_model=None, from_ckpt=False):
        """
        :param mode: TRAIN_MODE or TEST_MODE
        :param tc: a TrainConfig object or path to json format of the class
        :param ckpt_model: name of hdf5 format file present in config 'model_folder'
        :param from_ckpt: load model from checkpoint (for training only)
        """
        options = [TRAIN_MODE, TEST_MODE]
        assert mode in options, 'Invalid mode, choose from {}'.format(options)

        self.tc = load_train_config(tc)
        self.model_type = None
        self.k_model = None
        self.data, self.test_data = None, None
        self.test_x, self.test_y = None, None
        self.predicted = None
        self.evaluation = None
        self.ckpt_model = ckpt_model

        if mode == TRAIN_MODE:
            self.load_train_data()
            print(self.data.head())
            print('Train Data with shape {}'.format(self.data.shape))

            self.train_data_df, self.val_data_df = train_test_split(
                self.data, stratify=self.data.TARGET, random_state=tc.seed,
                test_size=self.tc.board_model['test_size']
            )

            self.x = self.train_data_df.drop('TARGET', axis=1).values
            self.y = self.train_data_df['TARGET'].values
            self.val_x = self.val_data_df.drop('TARGET', axis=1).values
            self.val_y = self.val_data_df['TARGET'].values.reshape((-1, 1))

            print('Train X: {} | Train Y: {} | Val X: {} | Val Y: {}'.format(self.x.shape, self.y.shape,
                                                                             self.val_x.shape, self.val_y.shape))

            self.class_weights = None
            if self.tc.use_class_weights:
                classes = [0., 1.]
                weights = compute_class_weight('balanced', classes, self.y)
                self.class_weights = dict(zip(classes, weights))
            self.y.reshape((-1, 1))
            if from_ckpt:
                self.k_model = load_model(
                    os.path.join(self.tc.board_model['model_folder'], ckpt_model),
                    custom_objects={'auc_roc': auc_roc}
                )
            else:
                self.k_model = self.build_model(self.tc.board_model['which'])
        else:
            self.k_model = load_model(
                os.path.join(self.tc.board_model['model_folder'], ckpt_model),
                custom_objects={'auc_roc': auc_roc}
            )

    def load_train_data(self):
        print('Loading train data')
        self.data = pd.read_csv(get_file('board_train_prob', self.tc.version), index_col='SK_ID_CURR')

    def load_test_data(self):
        print('Loading test data')
        self.test_data = pd.read_csv(get_file('board_test_prob', self.tc.version), index_col='SK_ID_CURR')
        print(self.test_data.head())
        print('Test Data with shape {}'.format(self.test_data.shape))

        self.test_x = self.test_data.drop('TARGET', axis=1).values
        self.test_y = self.test_data['TARGET'].values.reshape((-1, 1))

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

    def build_model(self, m):
        models = {
            0: self.model_0,
        }
        return models[m]()

    def get_model_folder(self, model_type=None, time=None):
        time_format = time or datetime.now().strftime('%m-%d_%H-%M-%S')
        model_type = model_type or self.model_type
        model_folder = '{type}_{time_format}'.format(type=model_type, time_format=time_format)
        return os.path.join(get_file('board_models', self.tc.version), model_folder)

    def get_callbacks(self):
        callbacks = []
        os.makedirs(get_file('board_models', self.tc.version), exist_ok=True)

        self.tc.board_model['model_folder'] = model_folder = self.get_model_folder()
        print('Model dir: {}'.format(model_folder))
        os.makedirs(model_folder, exist_ok=True)

        log_dir = os.path.join(model_folder, 'logs')
        print('Tensorboard log dir: {}'.format(log_dir))
        best_models_dir = os.path.join(model_folder, 'best')
        print('Best Model dir: {}'.format(best_models_dir))
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_models_dir, exist_ok=True)
        save_format = 'weights.{epoch:02d}-{val_loss:.2f}-{auc_roc:.2f}.hdf5'

        _j = os.path.join
        callbacks.append(ModelCheckpoint(_j(model_folder, save_format)))
        callbacks.append(ModelCheckpoint(_j(best_models_dir, save_format), save_best_only=True))
        callbacks.append(TensorBoard(log_dir=log_dir))

        return callbacks

    def model_0(self):
        self.model_type = 0
        bm = self.tc.board_model

        x = Input(shape=(self.x.shape[1],), name='board_inputs')
        hidden_layers = self.build_hidden_layers(
            x=x,
            layers=bm['layers'],
            params=bm['layer_params'],
            dropouts=bm['dropouts']
        )
        outputs = Dense(1, activation='sigmoid')(hidden_layers)

        model = Model(inputs=x, outputs=outputs)
        if bm['use_multi_gpu']:
            model = multi_gpu_model(model, gpus=bm['gpu_counts'])

        optimizer = OPTIMIZER_MAP[bm['optimizer']](**bm['optimizer_params'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy', auc_roc])
        return model

    def fit(self):
        bm = self.tc.board_model
        callbacks = self.get_callbacks()

        with open(os.path.join(self.tc.board_model['model_folder'], 'train_config.json'), 'w') as tcp:
            json.dump(json.loads(self.tc.to_json()), tcp)

        try:
            self.k_model.fit(
                x=self.x,
                y=self.y,
                batch_size=bm['batch_size'],
                epochs=bm['epochs'],
                callbacks=callbacks,
                validation_data=(self.val_x, self.val_y),
                class_weight=self.class_weights,
                # steps_per_epoch=train_steps,
                # validation_steps=val_steps,
                shuffle=True
            )
        except KeyboardInterrupt:
            print('Gracefully exiting training')

    def predict(self, x=None):
        if x is None and self.test_x is None:
            self.load_test_data()
            x = self.test_x
        else:
            x = x
        bm = self.tc.board_model
        self.predicted = self.k_model.predict(
            x=x,
            batch_size=bm['batch_size'],
        )

    def evaluate(self, x=None, y=None, update_rank=True):

        if x is None and self.data is None:
            self.load_train_data()

        x = x or self.data.drop('TARGET', axis=1).values
        y = y or self.data['TARGET'].values
        bm = self.tc.board_model
        self.evaluation = self.k_model.evaluate(
            x=x,
            y=y,
            batch_size=bm['batch_size'],
        )
        rank = dict(zip(self.k_model.metrics_names, self.evaluation))
        if update_rank:
            path = get_file('board_models_ranks', self.tc.version)
            try:
                with open(path) as fp:
                    current_rank = json.load(fp)
            except (json.JSONDecodeError, FileNotFoundError):
                current_rank = {}

            with open(path, 'w') as fp:
                current_rank[self.tc.board_model['model_folder']] = {
                    'rank': rank,
                    'model_ckpt': self.ckpt_model
                }
                json.dump(current_rank, fp)
        return rank

    def save_predicted(self, x=None, data=None, path=None, notes='', include_rank=True):
        os.makedirs(get_file('submission'), exist_ok=True)

        submission = pd.DataFrame()
        self.predict(x)
        rank = None if not include_rank else self.evaluate()
        data = data or self.test_data
        submission['SK_ID_CURR'] = data.index
        submission['TARGET'] = self.predicted

        file = 'submission_{:.4f}_{}.csv'.format(rank['auc_roc'], datetime.now().strftime('%m-%d_%H-%M-%S'))
        path = path or os.path.join(get_file('submission'), file)
        current_meta = safe_load_json(get_file('sub_meta'))
        sub_meta = {
            'path': path,
            'notes': notes,
            'model_path': self.tc.board_model['model_folder'],
            'model_ckpt': self.ckpt_model,
            'rank': rank
        }
        current_meta[file] = sub_meta
        dump_json(current_meta, get_file('sub_meta'))
        submission.to_csv(path, index=False)
