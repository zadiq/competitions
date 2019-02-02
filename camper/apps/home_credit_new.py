import numpy as np
import pandas as pd
from camper.new_models import (
    make_feature_selector,
    EstimatorStack, GroupExecutor,
    VotingEstimator
)
from camper.checkpoints import StatusLogCheckpoint
from camper.utils import folder_date
from skopt.space import Categorical, Integer
from sklearn.preprocessing import Imputer
import os

data_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/all.csv'
debug = False

if __name__ == '__main__':
    log_path = '/media/zadiq/ZHD/datasets/home_credit/models/{}'.format(folder_date())
    os.makedirs(log_path, exist_ok=True)
    meta = {
        'Name': 'Pima Diabetes',
        'Description': 'Testing the camper library',
    }
    logger = StatusLogCheckpoint(log_path, meta=meta)

    dataset = pd.read_csv(data_path, dtype='float32')
    dataset = dataset[dataset['TARGET'].notnull()]

    if debug:
        dataset = dataset.loc[:5000]

    features = [f for f in dataset.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    x, y = dataset[features], dataset['TARGET']
    imp = Imputer(np.nan, copy=False)
    x = x.replace([np.inf, -np.inf], np.nan)
    x[x.columns] = imp.fit_transform(x)
    x, y = x.values, y.values

    f_classif = make_feature_selector('f_classif', x, y)
    chi2 = make_feature_selector('mutual_info_classif', x, y)
    f_regression = make_feature_selector('f_regression', x, y)

    search_space_update = {
        'f_selector': Categorical([f_classif, chi2, f_regression]),
        'f_percentile': Integer(1, 100),
    }

    n_iter = 5 if debug else 50
    opt_params = {'cv': 10, 'n_iter': n_iter, 'n_points': 10}
    est_params = {
        'metric': 'roc_auc',
        'f_selector': f_classif,
        'search_space_update': search_space_update,
        'optimizer_params': opt_params,
        'logger': logger,
    }

    stack = EstimatorStack()
    stack = stack('kn_clf', **est_params)
    stack = stack('rf_clf', **est_params)
    # stack = stack('gp_clf', **est_params)
    stack = stack('gnb_clf', **est_params)
    stack = stack('et_clf', **est_params)
    stack = stack('gb_clf', **est_params)
    stack = stack('ab_clf', **est_params)
    stack = stack('lgbm_clf', **est_params)
    stack = stack('kn_clf', **est_params)
    stack = stack('qda_clf', **est_params)
    stack = stack('svc_clf', **est_params)

    ex = GroupExecutor(
        stack, logger, exe_params={'x': x, 'y': y},
        name='Optimization', num_of_workers=10
    )
    workers = ex.execute()
    workers.await(t=1)
    ex.update()

    voter = VotingEstimator(stack, logger=logger)
    voter.optimize(x, y)

    print('Finished!')
