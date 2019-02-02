import argparse
import sys
sys.path.append('/home/zoguntim/dev/home_credit_ml')
from credit.utils import timer
from credit.models import kfold_lightgbm, TrainConfig


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Home Credit')
    parser.add_argument("-c", "--config", help="path to configuration file", default=TrainConfig())

    parsed = parser.parse_args(sys.argv[1:])

    with timer("Train model"):
        data = kfold_lightgbm(tc=parsed.config)
    print('Finished!')


# python train.py -c /home/zoguntim/dev/home_credit_ml/runs/configs/cfg-2.json
