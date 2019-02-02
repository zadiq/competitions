import argparse
import sys
sys.path.append('/home/zoguntim/dev/home_credit_ml')
from credit.utils import timer
from credit.models import BoardModel, TrainConfig


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Home Credit')
    parser.add_argument("-c", "--config", help="path to configuration file", default=TrainConfig())

    parsed = parser.parse_args(sys.argv[1:])

    tc = TrainConfig()
    tc.board_model['layers'] = [50]
    tc.board_model['dropouts'] = [.4]
    tc.board_model['use_multi_gpu'] = False

    model = BoardModel(tc=tc)
    with timer("Train Board model"):
        model.fit()
    print('Finished!')


# python train_board_model.py -c /home/zoguntim/dev/home_credit_ml/runs/configs/cfg-2.json
