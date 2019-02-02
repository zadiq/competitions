import os
import datetime
import json
from keras.callbacks import (
    ReduceLROnPlateau, TensorBoard,
    ModelCheckpoint
)
from tgs import unpickle, pickle
from tgs.metrics import (
    iou, map_iou, get_map_loss,
    weight_loss_wrapper,
)


class TrainConfig:
    epochs = 200
    which_loss = 'map_iou'
    loss_at = False
    loss_log = False
    metrics = ["accuracy", iou, map_iou]
    log_base_dir = ''
    log_folder = None
    model_template = 'weights.{epoch:02d}-{val_loss:.2f}.model'
    save_best_only = True
    weight_wrapper_params = dict(
        protocol=0, pos_weight=.2,
        neg_weight=.8, pen_no_mask=False
    )
    unet_wrapper_params = dict(
        which='static', apply_threshold=False,
        input_shape=(128, 128, 1), unet_params=dict()
    )
    dataset_params = dict(
        validation_split=0.2,
        database_dir='/media/zadiq/ZHD/datasets/salt',
        img_shape=(101, 101, 1),
        train_img_shape=(128, 128, 1),
        seed=8090, extra_gen_params=dict(),
        extra_flow_params=dict()
    )
    meta = {
        'Description': 'A Salt Model',
        'Comments': ''
    }

    @classmethod
    def from_pickle(cls, path):
        return unpickle(path)

    @property
    def get_loss(self):
        if callable(self.which_loss):
            return self.which_loss
        return get_map_loss(self.which_loss, self.loss_at, self.loss_log)

    @property
    def get_wrapped_loss(self):
        return weight_loss_wrapper(self.get_loss, **self.weight_wrapper_params)

    @property
    def get_log_folder(self):
        if self.log_folder:
            return self.log_folder
        name = datetime.datetime.now().strftime('salt-%m-%d-%H-%M-%S')
        self.log_folder = os.path.join(self.log_base_dir, name)
        os.makedirs(self.log_folder)
        return self.log_folder

    @property
    def get_model_dir(self):
        folder = os.path.join(self.get_log_folder, 'models')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, self.model_template)
        print('Saving models to: {}'.format(folder))
        return path

    @property
    def get_tensorboard_dir(self):
        path = os.path.join(self.get_log_folder, 'logs')
        os.makedirs(path, exist_ok=True)
        print('Logging histories to: {}'.format(path))
        return path

    @property
    def get_callbacks(self):
        callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(self.get_model_dir, verbose=1, save_best_only=self.save_best_only),
            TensorBoard(self.get_tensorboard_dir),
        ]
        return callbacks

    def to_pickle(self, path=None):
        path = path or os.path.join(self.get_log_folder, 'train_config.pkl')
        print("pickling config to: {}".format(path))
        pickle(self, path)

    def save_meta(self):
        path = os.path.join(self.get_log_folder, 'meta.json')
        print("saving meta to: {}".format(path))
        with open(path, 'w') as fp:
            json.dump(self.meta, fp)

    def exit(self):
        self.to_pickle()
        self.save_meta()
