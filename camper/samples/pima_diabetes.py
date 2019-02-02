from numpy import loadtxt
from camper.new_models import (
    make_feature_selector,
    EstimatorStack, GroupExecutor,
    VotingEstimator
)
from camper.checkpoints import StatusLogCheckpoint
from camper.utils import folder_date
from skopt.space import Categorical, Integer
import os


if __name__ == '__main__':
    log_path = '/media/zadiq/ZHD/datasets/pima_diabetes/models/{}'.format(folder_date())
    os.makedirs(log_path, exist_ok=True)
    meta = {
        'Name': 'Pima Diabetes',
        'Description': 'Testing the camper library',
    }
    logger = StatusLogCheckpoint(log_path, meta=meta)

    dataset = loadtxt('/media/zadiq/ZHD/datasets/pima_diabetes/data.txt', delimiter=',')
    x, y = dataset[:, 0:8], dataset[:, 8]

    f_classif = make_feature_selector('f_classif', x, y)
    chi2 = make_feature_selector('chi2', x, y)
    search_space_update = {
        'f_selector': Categorical([f_classif, chi2]),
        'f_percentile': Integer(1, 100),
    }
    opt_params = {'cv': 50, 'n_iter': 5}
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
    stack = stack('gp_clf', **est_params)
    stack = stack('gnb_clf', **est_params)
    stack = stack('et_clf', **est_params)
    stack = stack('gb_clf', **est_params)
    stack = stack('ab_clf', **est_params)
    stack = stack('lgbm_clf', **est_params)
    stack = stack('kn_clf', **est_params)
    stack = stack('gnb_clf', **est_params)
    stack = stack('qda_clf', **est_params)
    stack = stack('svc_clf', **est_params)

    ex = GroupExecutor(
        stack, logger, exe_params={'x': x, 'y': y},
        name='Optimization', num_of_workers=len(stack)
    )
    workers = ex.execute()
    workers.await(t=1)
    ex.update()

    voter = VotingEstimator(stack, logger=logger)
    voter.optimize(x, y)

    print('Finished!')
