import os
import platform
import pandas as pd
import numpy as np
from camper.models import (
    MultiClassifierFeatureImportance, ClassifierStack,
    MultiClassifierOptimizer, VotingClassifier
)
from camper.variables import PROBA
from camper.checkpoints import StatusLogCheckpoint
from camper.utils import folder_date
from camper.search_space import (
    ET_CLF, RF_CLF, GB_CLF,
    AB_CLF, LGBM_CLF, SVC_CLF,
    KN_CLF, GP_CLF, GNB_CLF, QDA_CLF
)
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


data_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/all.csv'
log_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/voters_log/{}'.format(folder_date())
os.makedirs(log_path, exist_ok=True)
optimizer = BayesSearchCV
use_trace_callback = True
test_size = .30


if __name__ == '__main__':

    log_meta = {
        'Name': 'Home Credit Model',
        'Description': 'Voting ensemble of multiple classifier models'
    }
    logger = StatusLogCheckpoint(log_path, meta=log_meta)

    num_workers = 1
    num_of_points = 1
    if platform.node() in ['dgxadmin-DGX-Station']:
        num_workers = 10
        num_of_points = 10

    dataset = pd.read_csv(data_path, dtype='float32')
    dataset = dataset[dataset['TARGET'].notnull()]
    features = [f for f in dataset.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    x, y = dataset[features], dataset['TARGET']
    imp = Imputer(np.nan, copy=False)
    x = x.replace([np.inf, -np.inf], np.nan)
    x[x.columns] = imp.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

    classes = [0., 1.]
    weights = compute_class_weight('balanced', classes, y)
    class_weight = dict(zip(classes, weights))

    dt = DecisionTreeClassifier(class_weight=class_weight, max_depth=8)
    svc = SVC(kernel='linear', probability=True)  # docs claims it is too slow for large dataset
    ab_clf = AB_CLF.copy()
    ab_clf['base_estimator'] = Categorical([dt])

    lgbm_params = {'metric': 'auc', 'seed': 0}
    opt_params = {'refit': True, 'cv': 50, 'n_points': num_of_points}
    opt_ones_params = opt_params.copy()
    opt_ones_params['n_iter'] = 1
    stack_params = {
        'eval_metric': 'roc_auc',
        'opt_params': opt_params,
        'use_trace_callback': use_trace_callback,
        'logger': logger,
        'vote_space': Real(0, 10),
    }
    stack_one_params = stack_params.copy()
    stack_one_params['opt_params'] = opt_ones_params

    stack = ClassifierStack()
    stack = stack(ExtraTreesClassifier, search_space=ET_CLF, has_fi=True, **stack_params)
    stack = stack(RandomForestClassifier, search_space=RF_CLF, has_fi=True, **stack_params)
    stack = stack(GradientBoostingClassifier,  search_space=GB_CLF, has_fi=True, **stack_params)
    stack = stack(AdaBoostClassifier,  search_space=ab_clf, has_fi=True, **stack_params)
    stack = stack(LGBMClassifier, search_space=LGBM_CLF, params=lgbm_params, has_fi=True, **stack_params)
    stack = stack(KNeighborsClassifier, search_space=KN_CLF, **stack_params)
    stack = stack(SVC, search_space=SVC_CLF, params={'probability': True}, **stack_params)
    stack = stack(GaussianProcessClassifier, search_space=GP_CLF, **stack_params)
    stack = stack(GaussianNB, search_space=GNB_CLF, **stack_one_params)
    stack = stack(QuadraticDiscriminantAnalysis, search_space=QDA_CLF, **stack_params)

    mcfi = MultiClassifierFeatureImportance((x_train, y_train), (x_test, y_test), features=features,
                                            classifiers=stack.get_clf_with_fi(), raw_train_data=x,
                                            raw_target=y, eval_params={'eval_type': PROBA}, stratified=False,
                                            cross_val=True, num_of_workers=num_workers, logger=logger)
    mcfi.start()
    mcfi.workers.await()
    mcfi.log()

    mco = MultiClassifierOptimizer(optimizer, stack, (x, y), num_of_workers=num_workers, logger=logger)
    mco.start()
    mco.workers.await()
    mco.log()

    vc = VotingClassifier(stack, optimizer=optimizer, optimizer_params=opt_params, logger=logger)
    vc.optimize(x, y)
    vc.log()

    print('Finished!')
