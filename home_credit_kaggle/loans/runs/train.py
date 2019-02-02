from loans.models import BuildModel, TrainConfig
from loans.data import get_data_gen
from loans.utils import TRAIN_MODE, EVAL_MODE
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app_param = {
    'train_gen': get_data_gen(TRAIN_MODE, TrainConfig.batch_size),
    'eval_gen': get_data_gen(EVAL_MODE, TrainConfig.batch_size),
}
model = BuildModel(TRAIN_MODE, app_param)
