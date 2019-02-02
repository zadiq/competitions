import warnings
import time
import json
import queue
import traceback
import numpy as np
from camper.new_models.proxies import PROXY_MAP, EstimatorProxyBase
from camper.metrics import Metrics
from camper.workers import GroupWorkers
from sklearn.base import _pprint
from sklearn.feature_selection import (
    SelectPercentile, f_classif,
    f_regression, chi2, mutual_info_classif
)
from sklearn.externals import joblib as pickle
from skopt import BayesSearchCV
from skopt.space import Real


# TODO check that estimator type corresponds with scorer
FEATURE_SCORERS = {
    'f_classif': f_classif,
    'f_regression': f_regression,
    'chi2': chi2,
    'mutual_info_classif': mutual_info_classif,
}


def dummy(*args, **kwargs):
    return args, kwargs


def check_logger(logger):
    if logger == dummy:
        warnings.warn('Using a dummy logger, nothing will '
                      'be logged and saved.', UserWarning)


def make_feature_selector(scorer, x=None, y=None):

    msg = ('Invalid f_scorer provided, '
           'choose from {}.'.format(list(FEATURE_SCORERS.keys())))
    assert scorer in FEATURE_SCORERS, ValueError(msg)
    feature_selector = SelectPercentile(FEATURE_SCORERS[scorer])
    if x is not None and y is not None:
        return feature_selector.fit(x, y)
    return feature_selector


def load_metric(obj, params):
    if type(obj) == Metrics:
        return obj
    else:
        return Metrics(metric=obj, **params)


def load_proxy(obj):
    if issubclass(type(obj), EstimatorProxyBase):
        return obj

    not_str = type(obj) != str
    if not_str and issubclass(obj, EstimatorProxyBase):
        return obj()

    if not_str:
        raise ValueError('proxy argument should be '
                         'of type str or an EstimatorProxyBase subclass/object. '
                         '{} received.'.format(obj))

    try:
        return PROXY_MAP[obj]()
    except KeyError:
        raise ValueError("{} is not a valid proxy key, "
                         "choose from {}".format(obj, list(PROXY_MAP.keys())))


def load_est_from_json(path, _type='fitted'):
    with open(path) as fp:
        models = json.load(fp)
    uid = set([m.split('__')[0] for m in models])
    estimators = [pickle.load(models[u + '__' + _type]) for u in uid]
    return estimators


def load_est_from_path(path):
    return pickle.load(path)


class EstimatorBase:

    def __init__(self, proxy, metric, f_selector, proxy_klass_params=None,
                 metric_params=None, name=None, search_space_update=None,
                 f_percentile=100, optimizer_class=None, optimizer_params=None,
                 step=0, logger=dummy, last_ckpt_score=None, uid=None, weight=None,
                 early_stop_score=1):
        self.proxy_klass_params = proxy_klass_params or {}
        self.metric_params = metric_params or {}
        self.is_fitted = False
        self.logger = logger
        self.weight = weight or 1
        self.early_stop_score = early_stop_score
        check_logger(self.logger)

        self.optimizer = None
        self.optimizer_class = optimizer_class or BayesSearchCV
        self.optimizer_params = optimizer_params or {}

        self.proxy = load_proxy(proxy)
        self.metric = load_metric(metric, self.metric_params)
        self.scorer = None

        self.est = self.get_estimator
        self.name = name or self.proxy.klass.__name__
        self.has_feature_imp = self.proxy.has_feature_imp
        self.estimator_type = self.proxy.estimator_type
        self.search_space = self.proxy.search_space or {}
        self.search_space.update(search_space_update or {})

        self.f_selector = f_selector
        self.f_percentile = f_percentile

        self.step_counter = step
        self.last_ckpt_score = last_ckpt_score or 0
        self.uid = uid or str(int(time.time() * 1e7))

        self.str_id = self.name + '_' + self.uid

    def update_feature_selection(self):
        self.f_selector.percentile = self.f_percentile

    def callback(self, res):
        _ = res
        score = self.optimizer.best_score_

        self.logger(score, _type='score_trace', id=self.str_id)
        self.logger('{}: | step {} | score {}'.format(self.str_id,
                                                      self.step_counter,
                                                      score))

        if self._check_ckpt_score(score):
            self.save_model('ckpt')
        self.last_ckpt_score = score

        if score >= self.early_stop_score:
            return True

        self.step_counter += 1

    def save_model(self, _type='fitted'):
        filename = self.str_id + '_' + _type + '.pkl'
        self.logger(self, _type='model', file_name=filename, id=self.uid, name=self.name)

    def _check_ckpt_score(self, ckpt_score):
        # if self.metric.greater_is_better:
        # return max(self.last_ckpt_score, ckpt_score) != self.last_ckpt_score
        return ckpt_score > self.last_ckpt_score
        # return min(ckpt_score, ckpt_score) != self.last_ckpt_score

    def f_transform(self, x):
        """X transformation by feature selection"""
        assert hasattr(self.f_selector, 'scores_'), ValueError('Feature selection must be fitted')
        return self.f_selector.transform(x)

    def fit(self, x, y):
        x = self.f_transform(x)
        self.est.fit(x, y)

    def predict(self, x):
        x = self.f_transform(x)
        return self.est.predict(x)

    def predict_proba(self, x):
        x = self.f_transform(x)
        return self.est.predict_proba(x)

    def transform(self, x):
        return self.f_transform(x)

    def score(self, x, y):
        self.scorer = self.scorer or self.metric.make_scorer()
        return self.scorer(self, x, y)

    def optimize(self, x, y):
        if self.is_fitted:
            raise AttributeError('Estimator is already optimised')
        self.optimizer = self.optimizer_class(self, self.search_space, **self.optimizer_params)
        self.optimizer.fit(x, y, callback=self.callback)
        self.set_params(**self.optimizer.best_params_)
        self.est = self.optimizer.best_estimator_.est
        self.is_fitted = True
        self.save_model()
        self.logger({self.str_id: self.optimizer.best_score_}, _type='meta')

    @property
    def get_estimator(self):
        return self.proxy.klass(**self.proxy_klass_params)

    @property
    def estimator_params(self):
        return {
            'proxy': self.proxy,
            'metric': self.metric,
            'name': self.name,
            'f_selector': self.f_selector,
            'f_percentile': self.f_percentile,
            'search_space_update': None,
            'optimizer_params': self.optimizer_params,
            'step': self.step_counter,
            'last_ckpt_score': self.last_ckpt_score,
            'uid': self.uid,
            'weight': self.weight,
            'logger': self.logger,
        }

    def get_params(self, deep=False):
        _ = deep
        return self.estimator_params

    def set_params(self, **params):
        self.f_selector = params.pop('f_selector', self.f_selector)
        self.f_percentile = params.pop('f_percentile', self.f_percentile)
        self.update_feature_selection()
        self.proxy_klass_params.update(params)
        self.est.set_params(**params)
        return self

    def __str__(self):
        return "{}({})".format(self.name, _pprint(self.get_params(),
                                                  offset=len(self.name)))

    def __repr__(self):
        return self.__str__()


def check_estimator(est):
    assert (type(est) == EstimatorBase or
            issubclass(type(est), EstimatorBase)), ValueError('Expected estimator to be '
                                                              'of type EstimatorBase or its subclasses. '
                                                              'Got type {}'
                                                              .format(type(est)))


class EstimatorStack:

    def __init__(self):
        self.estimators = []

    @classmethod
    def from_json(cls, path, _type='fitted'):
        stack = cls()
        stack.estimators.extend(load_est_from_json(path, _type))
        return stack

    def append(self, est):
        check_estimator(est)
        self.estimators.append(est)

    def set_attr(self, attr, values):
        return [setattr(est, attr, val) for est, val in zip(self.estimators, values)]

    def get_attr(self, attr):
        return [getattr(est, attr) for est in self.estimators]

    def get_weights(self):
        return dict(zip(self.get_attr('uid'), self.get_attr('weight')))

    def update_weights(self, weights_dict):
        weights = [weights_dict[est.uid] for est in self.estimators]
        self.set_attr('weight', weights)

    @property
    def is_fitted(self):
        _bool = True
        for b in self.get_attr('is_fitted'):
            _bool &= b
        return _bool

    def est_is_unique(self, est):
        return est.uid not in self.get_attr('uid')

    def update(self, est):
        for i, e in enumerate(self.estimators):
            if e.uid == est.uid:
                self.estimators[i] = est
                return
        raise ValueError("Couldn't locate estimator with same uid {}.".format(est.uid))

    def __call__(self, *args, **kwargs):
        est = EstimatorBase(*args, **kwargs)
        if self.est_is_unique(est):
            self.append(est)
            return self
        # recreate a new estimator that generates a new unique id
        return self(*args, **kwargs)

    def __getitem__(self, item):
        return self.estimators[item]

    def __iter__(self):
        yield from self.estimators

    def __len__(self):
        return len(self.estimators)

    def __str__(self):
        return 'EstimatorStack({})'.format(len(self.estimators))

    def __repr__(self):
        return self.__str__()


class GroupExecutor:

    def __init__(self, estimators, logger=dummy, num_of_workers=5,
                 execution='optimize', exe_params=None, name=None):
        self.estimators = estimators
        self.logger = logger
        check_logger(self.logger)

        self.num_of_workers = num_of_workers
        self.use_process = True

        self.execution = execution
        self.exe_params = exe_params or {}

        self.name = name or execution.capitalize()
        self.has_finished = []
        self.gw = self.get_workers

    @property
    def get_workers(self):
        return GroupWorkers(
            self.num_of_workers, self._execute,
            self.estimators, name=self.name,
            use_process=self.use_process
        )

    def execute(self):
        self.logger('{}: starting'.format(self.name))
        self.gw.start()
        return self.gw

    def _execute(self, estimators, worker):
        num_of_tasks = len(estimators)
        for ix, est in enumerate(estimators):
            str_id = "{}:{}({}/{})".format(worker, est.str_id, ix+1, num_of_tasks)
            self.logger('{}: started working on task {}/{}|{}'.format(
                worker, ix+1, num_of_tasks,  est.str_id)
            )
            try:
                getattr(est, self.execution)(**self.exe_params)
                worker.queue.put(est)
            except Exception as e:
                worker.error = e
                worker.error_occurred = True
                worker.finished_event.set()
                self.logger('{}: error occurred. \n{}'.format(str_id, traceback.format_exc()))
                raise e

        worker.finished_event.set()
        self.logger('{}: has finished'.format(worker))

    def update(self):
        for worker in self.gw.workers.values():
            while True:
                try:
                    self.estimators.update(worker.queue.get(timeout=1))
                except queue.Empty:
                    break


class VotingEstimator:

    def __init__(self, stack: EstimatorStack, optimizer_class=None, opt_params=None,
                 voting='soft', logger=dummy, name='VotingEstimator', early_stop_score=1.,
                 scorer=None, step_counter=0):

        self.stack = stack
        assert self.stack.is_fitted, ValueError('All estimators in stack should be fitted')
        self.optimizer_class = optimizer_class or BayesSearchCV
        self.opt_params = opt_params or {}
        self.voting = voting
        self.logger = logger
        check_logger(self.logger)
        self.name = name
        self.early_stop_score = early_stop_score
        self.optimizer = None
        self.scorer = scorer or self.stack[0].metric.make_scorer()
        self.step_counter = step_counter
        self.is_optimised = False
        self.stopped_early = False

    @property
    def weights(self):
        return self.stack.get_attr('weight')

    def _predict(self, x):
        return np.array([est.predict(x) for est in self.stack]).T

    def _collect_proba(self, x):
        return np.array([clf.predict_proba(x) for clf in self.stack])

    def _callback(self, res):

        _ = res
        score = self.optimizer.best_score_

        self.logger(score, _type='score_trace', id=self.name)
        self.logger('{}: | step {} | score {}'.format(self.name,
                                                      self.step_counter,
                                                      score))
        self.logger(self, _type='voter')
        if score >= self.early_stop_score:
            self.stopped_early = True
            return True

        self.step_counter += 1

    def fit(self, x, y):
        """Do nothing since the estimators are already fitted"""

    def predict(self, x):
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(x), axis=1)
        else:
            predictions = self._predict(x)
            maj = np.apply_along_axis(lambda _x: np.argmax(np.bincount(_x, weights=self.weights)),
                                      axis=1, arr=predictions.astype('int'))
        return maj

    def predict_proba(self, x):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when voting={}".format(self.voting))

        return np.average(self._collect_proba(x), axis=0, weights=self.weights)

    def optimize(self, x, y):
        self.optimizer = self.optimizer_class(self, self._get_vote_spaces, **self.opt_params)
        self.optimizer.fit(x, y, callback=self._callback)
        self.stack.update_weights(self.optimizer.best_params_)
        self.is_optimised = True

        log = {
            self.name: {
                'best_score': self.optimizer.best_score_,
                'best_params': self.optimizer.best_params_,
                'weights': self.weights,
                'stopped_early': self.stopped_early,
                'num_of_steps': self.step_counter,
            }
        }
        self.logger(log, _type='vc')
        self.logger(self, _type='voter')
        self.logger('Finished Optimising VotingEstimator with '
                    'a score of {}'.format(self.optimizer.best_score_))

    def transform(self, x):
        if self.voting == 'soft':
            return self._collect_proba(x)
        else:
            return self._predict(x)

    def score(self, x, y):
        return self.scorer(self, x, y)

    @property
    def _get_vote_spaces(self):
        _len = len(self.stack)
        return dict(zip(self.stack.get_attr('uid'), [Real(0, _len)] * _len))

    def get_params(self, deep=True):
        _ = deep
        return {
            'stack': self.stack,
            'voting': self.voting,
            'logger': self.logger,
            'name': self.name,
            'early_stop_score': self.early_stop_score,
            'scorer': self.scorer,
            'step_counter': self.step_counter,
        }

    def set_params(self, **params):
        self.stack.update_weights(params)

    def __str__(self):
        return "{}({})".format(self.name, _pprint(self.get_params(),
                                                  offset=len(self.name)))

    def __repr__(self):
        return self.__str__()
