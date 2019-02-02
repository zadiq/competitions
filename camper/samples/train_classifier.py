import os
import pandas as pd
import platform
from numpy import loadtxt
from camper.models import (
    MultiClassifierFeatureImportance, ClassifierStack,
    MultiClassifierOptimizer, VotingClassifier
)
from camper.variables import PROBA
from camper.checkpoints import StatusLogCheckpoint
from camper.utils import folder_date
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


# SEARCH SPACE
FOREST_TREE = {
        'n_estimators': Integer(4, 10)
    }
G_BOOST = {
    'learning_rate': Real(.1, .3)
}
A_BOOST = {
    'learning_rate': Real(.8, 1.)
}
KN_CLF = {
    'n_neighbors': Integer(5, 10),
}
SVC_CLF = {
    'C': Real(0.001, 5),
}
GP_CLF = {
    'kernel': Categorical([None])
}
MLP_CLF = {
    'hidden_layer_sizes': Categorical([(100, 100, 200), (200, 100, 200), (300, 200, 100)])
}
GNB_CLF = {
    'priors': Categorical([None])
}
QDA_CLF = {
    'reg_param': Real(0, 1)
}

log_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/voters_log/{}'.format(folder_date())
os.makedirs(log_path, exist_ok=True)
use_trace_callback = True


if __name__ == '__main__':
    dataset = loadtxt('./data/pima_diabetes.txt', delimiter=',')
    x, y = dataset[:, 0:8], dataset[:, 8]
    test_size = .30
    train_data, target = pd.DataFrame(x), pd.DataFrame(y)
    features = train_data.columns.tolist()
    x, y = train_data[features], target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

    log_meta = {
        'Name': 'Pima Diabetes Model',
        'Description': 'Testing camper'
    }
    logger = StatusLogCheckpoint(log_path, meta=log_meta)

    optimizer = BayesSearchCV
    pre_dispatch = 1 if platform.node() == 'zadiq_linux' else '2*n_jobs'
    opt_params = {'refit': True, 'cv': 5, 'n_jobs': 1}
    opt_ones_params = opt_params.copy()
    opt_ones_params['n_iter'] = 1
    stack_params = {
        'eval_metric': 'roc_auc',
        'opt_params': opt_params,
        'use_trace_callback': use_trace_callback,
        'logger': logger,
        'vote_space': Real(1, 5),
    }
    stack_one_params = stack_params.copy()
    stack_one_params['opt_params'] = opt_ones_params

    stack = ClassifierStack()
    stack = stack(ExtraTreesClassifier, search_space=FOREST_TREE, has_fi=True, **stack_params)
    stack = stack(RandomForestClassifier, search_space=FOREST_TREE, has_fi=True, **stack_params)
    stack = stack(GradientBoostingClassifier,  search_space=G_BOOST, has_fi=True, **stack_params)
    stack = stack(AdaBoostClassifier,  search_space=A_BOOST, has_fi=True, **stack_params)
    stack = stack(LGBMClassifier, search_space=G_BOOST, has_fi=True, **stack_params)
    stack = stack(KNeighborsClassifier, search_space=KN_CLF, **stack_params)
    stack = stack(SVC, search_space=SVC_CLF, params={'probability': True}, **stack_params)
    stack = stack(GaussianProcessClassifier, search_space=GP_CLF, **stack_one_params)
    # stack = stack(MLPClassifier, search_space=MLP_CLF, params={'activation': 'logistic'}, **stack_params)
    stack = stack(GaussianNB, search_space=GNB_CLF, **stack_one_params)
    stack = stack(QuadraticDiscriminantAnalysis, search_space=QDA_CLF, **stack_params)

    mcfi = MultiClassifierFeatureImportance((x_train, y_train), (x_test, y_test), features=features,
                                            classifiers=stack.get_clf_with_fi(), raw_train_data=train_data,
                                            raw_target=target, eval_params={'eval_type': PROBA}, stratified=False,
                                            cross_val=True, num_of_workers=5, logger=logger)
    mcfi.start()
    mcfi.workers.await()
    mcfi.log()

    mco = MultiClassifierOptimizer(optimizer, stack, (x, y), num_of_workers=3, logger=logger)
    mco.start()
    mco.workers.await()
    mco.log()

    vc = VotingClassifier(stack, optimizer=optimizer, optimizer_params=opt_params, logger=logger)
    vc.optimize(x, y)
    vc.log()

    print('Finished!')
