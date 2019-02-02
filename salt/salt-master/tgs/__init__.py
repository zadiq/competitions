import os
import platform
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from tqdm import tqdm_notebook as tn
from tgs.metrics import map_iou
import gc
import dill


load_keras_model = load_model
BASE_DIR = '/media/zadiq/ZHD/datasets/salt'


def allocate_gpu(dgx=False):

    if platform.node() != "zadiq-linux":
        while True:
            if dgx:
                result = subprocess.run(["/AvailableGPU/available_gpu", "-m", "15000"], stdout=subprocess.PIPE)
            else:
                result = subprocess.run(["/AvailableGPU/available_gpu", "-m", "10000"], stdout=subprocess.PIPE)
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(result.stdout))
                print("Memory to be allocated in GPU ", str(int(result.stdout)))
                break
            except:
                print("No GPU available! Waiting (10 seconds)...")
                time.sleep(10)


def unpickle(path):
    with open(path, 'rb') as fp:
        return dill.load(fp)


def pickle(obj, path):
    with open(path, 'wb') as fp:
        return dill.dump(obj, fp)


def get_model(folder, which, custom_objects=None, loss_name="", loss_is_wrapped=True,
              base_dir=BASE_DIR, train_config=None):
    _ = os.path.join
    model_base_dir = _(base_dir, 'models', folder)
    T = train_config if train_config else unpickle(_(model_base_dir, 'train_config.pkl'))
    custom_objects = custom_objects or {}

    if loss_name:
        custom_objects[loss_name] = T.get_wrapped_loss if loss_is_wrapped else T.get_loss

    return load_model(_(model_base_dir, 'models', which), custom_objects=custom_objects)


def choose_threshold(model, val_gen, metric=map_iou, n=50):

    thresholds = np.arange(0, 1.0, 0.1)
    scores = []

    for _ in tn(range(n)):
        val_batch = next(val_gen)
        val_batch_pred = model.predict(val_batch[0])
        batch_scores = []

        for i in thresholds:
            pred_at = (val_batch_pred > i).astype('float32')
            batch_scores.append(K.eval(metric(val_batch[1], pred_at)))

        scores.append(batch_scores)

        del val_batch
        gc.collect()

    scores = np.array(scores).mean(axis=0)
    best_threshold = thresholds[scores.argmax()]

    plt.plot(thresholds, scores)
    plt.plot([best_threshold], [scores.max()], marker='x', label="Best Threshold: {}".format(best_threshold))
    plt.legend()

    return best_threshold
