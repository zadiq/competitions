from camper.models import ClassifierStack, VotingClassifier
from camper.checkpoints import StatusLogCheckpoint
from skopt import BayesSearchCV
from numpy import loadtxt
import pandas as pd


if __name__ == '__main__':
    dataset = loadtxt('./data/pima_diabetes.txt', delimiter=',')
    x, y = dataset[:, 0:8], dataset[:, 8]
    test_size = .30
    train_data, target = pd.DataFrame(x), pd.DataFrame(y)
    features = train_data.columns.tolist()
    x, y = train_data[features], target

    optimizer = BayesSearchCV
    opt_params = {'refit': True, 'cv': 5}

    logger = StatusLogCheckpoint('C:\\Users\\genstry\\dev\\camper_tools\\logs')
    stack = ClassifierStack()
    stack.from_json('C:\\Users\\genstry\\dev\\camper_tools\\logs\\models.json')

    vc = VotingClassifier(stack, optimizer=optimizer, optimizer_params=opt_params, logger=logger)
    vc.optimize(x, y)
    vc.log()
