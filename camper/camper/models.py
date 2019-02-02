import gc
import time
import json
import numpy as np
import pandas as pd
import warnings
from camper.metrics import Metrics
from camper.variables import PREDICT, PROBA, CLF_WITH_FI
from camper.workers import GroupWorkers
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer
from sklearn.externals import joblib as pickle


warnings.filterwarnings(action='ignore', category=UserWarning)


def dummy(*args, **kwargs):
    return args, kwargs


def check_logger(logger):
    if logger == dummy:
        warnings.warn('Using a dummy logger, nothing will be logged and saved.', UserWarning)


def load_models(path, _type='fitted'):
    with open(path) as fp:
        models = json.load(fp)
    uids = set([m.split('__')[0] for m in models])
    clfs = [pickle.load(models[u + '__' + _type]) for u in uids]
    return clfs


def check_has_fi(clf):
    if not (hasattr(clf, 'has_fi') or type(clf) in CLF_WITH_FI):
        raise ValueError('{} does not have/support feature importances.'.format(clf))


class ClassifierStack:

    def __init__(self):
        self.classifiers = []

    def check_uniqueness(self):
        for clf in self.classifiers:
            if not self.clf_is_unique(clf):
                return False
        return True

    @property
    def is_fitted(self):
        for clf in self.classifiers:
            if not clf.best_estimator_:
                return False
        return True

    def from_json(self, path, _type='fitted'):
        self.classifiers.extend(load_models(path, _type))
        if not self.check_uniqueness():
            warnings.warn('Classifiers in stack are not unique')

    def from_list(self, obj):
        self.classifiers.extend(obj)

    def reset(self):
        [c.reset() for c in self.classifiers]

    def get_attr(self, attr):
        return [getattr(clf, attr) for clf in self.classifiers]

    def get_uid(self):
        return self.get_attr('uid')

    def get_clf_by_id(self, uid):
        for clf in self.classifiers:
            if clf.uid == uid:
                return clf

    def get_clf_with_fi(self):
        # get classifiers with feature importance
        return [clf for clf in self.classifiers if clf.has_fi]

    def get_vote_weights(self):
        return self.get_attr('vote_weight')

    def set_vote_weights(self, weights):
        for uid, w in weights.items():
            clf = self.get_clf_by_id(int(uid))
            clf.vote_weight = w

    def clf_is_unique(self, clf):
        return clf.uid not in self.get_uid()

    def get_vote_grid(self):
        grid = dict(zip(map(str, self.get_uid()), self.get_attr('vote_space')))
        grid.update({'classifiers': [self]})
        return grid

    def __call__(self, clf, **kwargs):
        clf_pro = ClassifierProxy(clf, **kwargs)
        if not self.clf_is_unique(clf_pro):
            return self.__call__(clf, **kwargs)
        self.classifiers.append(clf_pro)
        return self

    def __iter__(self):
        yield from self.classifiers

    def __len__(self):
        return len(self.classifiers)

    def __getitem__(self, item):
        return self.classifiers[item]

    def __str__(self):
        return 'ClassifierStack({}) <{} ...>'.format(len(self.classifiers), self.classifiers[:3])

    def __repr__(self):
        return self.__str__()


class ClassifierProxy:

    def __init__(self, clf, eval_metric, params=None, name=None, opt_params=None, vote_space=None,
                 reverse_scores=True, scorer=None, search_space=None, opt_fit_params=None,
                 use_trace_callback=False, logger=dummy, imp_scale=10, has_fi=False):
        self.clf_class = clf
        self.name = name or clf.__name__
        self.clf_params = params or {}
        self.clf = self.clf_class(**self.clf_params)
        self.logger = logger
        check_logger(self.logger)
        self.reverse_scores = reverse_scores
        self.ckpt_counter = 0
        self.vote_space = vote_space
        self.vote_weight = 1
        self.imp_scale = imp_scale
        if has_fi:
            check_has_fi(self.clf)
        self.has_fi = has_fi

        self.features_imp = pd.DataFrame()
        self.feature_thresholds = {}
        self.best_threshold = {}
        self.selections = {}
        self.best_selection_cache = None

        self.eval_metric = Metrics(eval_metric)
        self.eval_type = self.eval_metric.eval_type
        self.evaluated = False
        needs_proba = True if self.eval_type == PROBA else False
        self.scorer = scorer or make_scorer(self.eval_metric.scorer(), greater_is_better=self.reverse_scores,
                                            needs_proba=needs_proba)

        self.search_space = search_space or {}
        self.opt_params = opt_params or {}
        self.opt_clf_score = None
        self.opt_ckpt_best_score = 0
        self.opt_fit_params = opt_fit_params or {}
        self.opt_clf = None
        self.opt_trace = []
        self.best_estimator_ = None
        if use_trace_callback:
            # self.trace_callback = trace_callback(self)
            self.opt_fit_params.update({'callback': self.trace_callback})

        self.uid = int(time.time() * 1e7)

    def trace_callback(self, res):
        _ = res
        self.opt_trace.append(self.opt_clf.best_score_)
        log = {self.name: self.opt_trace}
        self.logger(log, _type='trace')
        if self.ckpt_score_is_better():
            self.logger('{}: saving at ckpt:{}, score:{}'.format(self.name, self.ckpt_counter,
                                                                 self.opt_clf.best_score_))
            self.logger(self, _type='model', file_name=self.filename('ckpt'), name='ckpt', id=self.uid)
        else:
            self.logger('{}: NOT saving at ckpt:{}, score:{}'.format(self.name, self.ckpt_counter,
                                                                     self.opt_clf.best_score_))
        self.opt_ckpt_best_score = self.opt_clf.best_score_
        self.ckpt_counter += 1

    def ckpt_score_is_better(self):
        if self.reverse_scores:
            return max(self.opt_ckpt_best_score, self.opt_clf.best_score_) != self.opt_ckpt_best_score
        return min(self.opt_clf.best_score_, self.opt_ckpt_best_score) != self.opt_ckpt_best_score

    def filename(self, app=''):
        return self.name + '_' + app + '.pkl'

    def reset(self, params=None):
        params = params or self.clf_params
        self.clf = self.clf_class(**params)
        self.evaluated = False

    def fit(self, train, imp=False, features=None, n_fold=0):
        self.clf.fit(train[0], train[1])

        if imp:
            assert features, 'features must be provided when computing importance'
            n_imp_df = pd.DataFrame()
            n_imp_df['features'] = features
            n_imp_df['fold'] = n_fold
            n_imp_df['importance'] = self.clf.feature_importances_
            self.features_imp = pd.concat([self.features_imp, n_imp_df], axis=0)

    def transform(self, x):
        return self.best_selection.transform(x) if self.has_fi else x

    def predict_proba(self, x):
        return self.opt_clf.predict_proba(self.transform(x))

    def predict(self, x):
        return self.opt_clf.predict(self.transform(x))

    def eval(self, eval_data, eval_type=None, compute_metric=False, order=None):
        pred = None
        order = order or [0, 1]

        eval_type = eval_type or self.eval_type

        if eval_type == PREDICT:
            pred = self.clf.predict(eval_data[0])
        elif eval_type == PROBA:
            pred = self.clf.predict_proba(eval_data[0])[:, 1]

        if compute_metric:
            tp = [eval_data[1], pred]
            self.eval_metric(tp[order[0]], tp[order[1]])
            self.evaluated = True

        return pred

    @property
    def score(self):
        if not self.evaluated:
            raise ValueError('Classifier must be evaluated first by calling `eval` method')
        return self.eval_metric.score

    def hash_best_score(self, h):
        return self.feature_thresholds[h][self.best_threshold[h]]

    @property
    def best_hash(self):
        return sorted(self.best_threshold, key=self.hash_best_score, reverse=self.reverse_scores)[0]

    @property
    def best_score(self):
        return self.hash_best_score(self.best_hash)

    @property
    def best_selection(self):
        if self.best_selection_cache:
            return self.best_selection_cache
        self.best_selection_cache = self.selections[self.best_hash][self.best_threshold[self.best_hash]]
        self.selections = {}

    def log(self, others=None):
        print('{}: | {}: {:.4f} | {}'.format(self.name, self.eval_metric.name, self.score, others))

    def log_mcfi(self):
        print('{}<{}>: (Best Score: {:.4f} | Threshold: {:.4f})'.format(self.name, self.best_hash, self.best_score,
                                                                        self.best_threshold[self.best_hash]))

    def log_mco(self):
        self.logger(self, _type='model', file_name=self.filename('fitted'), name='fitted', id=self.uid)
        print('{}: (Train Score: {:.4f} | Test Score: {:.4f})'.format(self.name, self.opt_clf.best_score_,
                                                                      self.opt_clf_score))

    def agg_imp(self):
        """Aggregate feature importance"""
        return self.features_imp.groupby('features').mean().sort_values(by='importance')

    def __str__(self):
        return "ClassifierProxy({})".format(self.name)

    def __repr__(self):
        return self.__str__()


class MultiClassifierFeatureImportance:
    """Perform feature importance on a data with chosen classifier(s)"""

    def __init__(self, train_data, test_data, classifiers, features, raw_train_data=None, raw_target=None,
                 random_state=1, cross_val=True, stratified=True, n_folds=5, shuffle=True, eval_params=None,
                 test_size=.30, num_of_workers=1, logger=dummy):

        for clf in classifiers:
            if not clf.has_fi:
                raise ValueError('MultiClassifierFeatureImportance only accepts classifiers with feature importance')

        self.is_computed = False
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        self.classifiers = classifiers
        self.raw_train_data = raw_train_data
        self.raw_target = raw_target
        self.features = features
        self.cross_val = cross_val
        self.random_state = random_state
        self.stratified = stratified
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.eval_params = eval_params or {}
        self.eval_params.update({'compute_metric': True})
        self.test_size = test_size
        self.hash_key = 'HASH_KEY_{}_{}_{}'.format(stratified, cross_val, n_folds)
        self.logger = logger
        check_logger(self.logger)

        self.workers = GroupWorkers(num_of_workers, self.compute, self.classifiers, name='FeatureImp')
        self.logger('MultiClassifierFeatureImportance: Created instance')

    def start(self):
        self.logger('MultiClassifierFeatureImportance: Started workers')
        self.workers.start()

    def log(self):
        print('\n-------------------------------------------------------------------------------------')
        for clf in self.classifiers:
            clf.log_mcfi()
        print('-------------------------------------------------------------------------------------\n')
        self.logger('MultiClassifierFeatureImportance: finished featuree importance selection')

    def compute(self, classifiers, worker):
        try:
            for clf in classifiers:

                if self.cross_val:
                    assert (self.raw_train_data is not None and
                            self.raw_target is not None), ('Raw training data and target '
                                                           'must be provided when using '
                                                           'cross validation!')

                    if self.stratified:
                        folds = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle,
                                                random_state=self.random_state)
                    else:
                        folds = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                    t = self.raw_target.values.ravel()
                    for n_fold, (train_ix, valid_ix) in enumerate(folds.split(self.raw_train_data[self.features], t)):
                        x, y = self.raw_train_data[self.features].iloc[train_ix], self.raw_target.iloc[train_ix]
                        x_val, y_val = self.raw_train_data[self.features].iloc[valid_ix], self.raw_target.iloc[valid_ix]
                        clf.fit((x, y.values.ravel()), imp=True, features=self.features, n_fold=n_fold)
                        clf.eval((x_val, y_val), **self.eval_params)
                        clf.log('Fold: {} |'.format(n_fold))
                        clf.reset()
                else:
                    clf.fit((self.x_train, self.y_train), imp=True, features=self.features)
                    clf.eval((self.x_test, self.y_test), **self.eval_params)
                    clf.log()

                clf.reset()
                agg = clf.agg_imp()
                clf.feature_thresholds[self.hash_key] = {}
                clf.selections[self.hash_key] = {}
                importance = agg['importance']
                _len = len(importance)
                for ix in range(0, _len, (_len//clf.imp_scale or 1)):
                    imp = importance[ix]
                    selection = SelectFromModel(clf.clf, threshold=imp)
                    select_x_train = selection.fit_transform(self.x_train, self.y_train.values.ravel())
                    select_x_test = selection.transform(self.x_test)
                    if select_x_train.shape[1]:
                        clf.fit((select_x_train, self.y_train.values.ravel()))
                        clf.eval((select_x_test, self.y_test), **self.eval_params)
                        clf.log('No of Features: {} |'.format(select_x_train.shape[1]))
                        clf.feature_thresholds[self.hash_key][imp] = clf.score
                        clf.selections[self.hash_key][imp] = selection
                    clf.reset()

                    del select_x_train, select_x_test
                    gc.collect()
                # TODO choose threshold based on both score and least num of features
                clf.best_threshold[self.hash_key] = sorted(clf.feature_thresholds[self.hash_key],
                                                           key=clf.feature_thresholds[self.hash_key].get,
                                                           reverse=clf.reverse_scores)[0]
                best_score = clf.best_score
                _ = clf.best_selection
                print('{}<{}>: (Best Score: {:.4f} | Threshold: {:.4f})'.format(clf.name, self.hash_key, best_score,
                                                                                clf.best_threshold[self.hash_key]))
                log = {
                    clf.name: {
                        'best_hash': clf.best_hash,
                        'best_score': clf.best_score,
                        'best_threshold': clf.best_threshold[clf.best_hash]
                    }
                }
                self.logger(log, _type='mcfi')
                clf.reset()
        except Exception:
            worker.finished = True
            raise

        self.logger('MultiClassifierFeatureImportance: {} has finished'.format(worker))
        worker.finished = True


class MultiClassifierOptimizer:

    def __init__(self, optimizer, classifiers, train_data, num_of_workers=1, logger=dummy):
        self.optimizer = optimizer
        self.classifiers = classifiers
        self.x_train, self.y_train = train_data
        self.logger = logger
        check_logger(self.logger)

        self.workers = GroupWorkers(num_of_workers, self.optimize, self.classifiers, name='Optimizer')
        self.logger('MultiClassifierOptimizer: Created instance')

    def start(self):
        self.logger('MultiClassifierOptimizer: Started workers')
        self.workers.start()

    def optimize(self, classifiers, worker):
        for clf in classifiers:
            try:
                self.logger('{}: Started optimization'.format(clf.name))
                clf.reset()
                select_x_train = clf.transform(self.x_train)
                clf.opt_clf = self.optimizer(clf.clf, clf.search_space, scoring=clf.scorer, **clf.opt_params)
                clf.opt_clf.fit(select_x_train, self.y_train.values.ravel(), **clf.opt_fit_params)
                clf.opt_clf_score = clf.opt_clf.score(select_x_train, self.y_train.values.ravel())
                clf.best_estimator_ = clf.opt_clf.best_estimator_
                log = {
                    clf.name: {
                        'opt_best_score': clf.opt_clf.best_score_,
                        'train_set_score': clf.opt_clf_score,
                        'num_iterations': clf.ckpt_counter
                    }
                }
                self.logger(log, _type='mco')
                del select_x_train
                gc.collect()
            except Exception as e:
                self.logger('{}: Error occurred. \n {}'.format(clf.name, e))
                worker.error = e
                worker.error_occurred = True
                raise

        self.logger('MultiClassifierOptimizer: {} has finished'.format(worker))
        worker.finished = True

    def log(self):
        print('\n-------------------------------------------------------------------------------------')
        print('Optimized Models')
        for clf in self.classifiers:
            clf.log_mco()
        print('-------------------------------------------------------------------------------------\n')
        self.logger('MultiClassifierOptimizer: finished optimizing model')


def _get_params(params):
    def get(deep):
        _ = deep
        return params
    return get


class VotingClassifier:

    def __init__(self, classifiers: ClassifierStack, optimizer=None, optimizer_params=None,
                 voting='soft', logger=dummy, name='VotingClassifier', early_stop_score=1.):

        if not classifiers.is_fitted:
            raise ValueError('VotingClassifier only accepts fitted classifiers')

        self.name = name
        self.classifiers = classifiers
        self.voting = voting
        self.scorer = self.classifiers[0].scorer
        self.logger = logger
        check_logger(self.logger)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params or {}
        self.opt_voter = None
        self.is_optimized = False
        self.stopped_early = False
        self.best_weights_ = None
        self.best_estimator_ = None
        self.early_stop_score = early_stop_score
        self._trace = []

        if self.optimizer:  # else it probably being cloned by optimizer
            self.logger('VotingClassifier: created instance')

    @classmethod
    def from_pkl(cls, path):
        return pickle.load(path)

    @property
    def weights(self):
        return self.classifiers.get_vote_weights()

    def get_params(self, deep=False):
        _ = deep
        params = {
            'name': self.name,
            'classifiers': self.classifiers,
            'voting': self.voting,
            'logger': self.logger,
            'optimizer_params': self.optimizer_params,
            'early_stop_score': self.early_stop_score,
        }
        return params

    def _callback(self, res):
        _ = res
        self._trace.append(self.opt_voter.best_score_)
        self.logger({self.name: self._trace}, _type='trace')
        if self.opt_voter.best_score_ >= self.early_stop_score:
            self.stopped_early = True
            return True

    def set_params(self, **params):
        params.pop('classifiers')
        self.classifiers.set_vote_weights(params)
        return self

    def optimize(self, x, y):
        if not self.optimizer:
            raise AttributeError('optimize method is not available when optimizer is not provided')
        self.logger('VotingClassifier: started optimizer')
        self.opt_voter = self.optimizer(self, self.classifiers.get_vote_grid(), **self.optimizer_params)
        self.opt_voter.fit(x, y, callback=self._callback)
        self.best_weights_ = dict(zip(self.classifiers.get_uid(), self.opt_voter.best_estimator_.weights))
        self.classifiers.set_vote_weights(self.best_weights_)
        self.opt_voter.best_estimator_.optimizer = self.optimizer
        self.best_estimator_ = self.opt_voter.best_estimator_
        self.is_optimized = True

    def log(self):
        log = {
            self.name: {
                'best_score': self.opt_voter.best_score_,
                'best_params': self.opt_voter.best_params_,
                'best_weights': self.best_weights_,
                'stopped_early': self.stopped_early,
            }
        }
        log[self.name]['best_params'].pop('classifiers')
        self.logger('VotingClassifier: finished optimizing')
        self.logger(log, _type='vc')
        self.logger(self, _type='voter')

    def score(self, x, y):
        return self.scorer(self, x, y)

    def _collect_proba(self, x):
        return np.array([clf.predict_proba(x) for clf in self.classifiers])

    def fit(self, x, y, **kwargs):
        """Do nothing since the estimators are already fitted"""

    def _predict(self, x):
        return np.array([clf.predict(x) for clf in self.classifiers]).T

    def predict_proba(self, x):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when voting={}".format(self.voting))
        return np.average(self._collect_proba(x), axis=0, weights=self.weights)

    def predict(self, x):
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(x), axis=1)
        else:
            predictions = self._predict(x)
            maj = np.apply_along_axis(lambda _x: np.argmax(np.bincount(_x, weights=self.weights)),
                                      axis=1, arr=predictions.astype('int'))
        return maj

    def transform(self, x):
        if self.voting == 'soft':
            return self._collect_proba(x)
        else:
            return self._predict(x)

    def __str__(self):
        params = self.get_params()
        params.update(self.best_weights_ or {})
        return 'VotingClassifier({})'.format(params)

    def __repr__(self):
        return self.__str__()
