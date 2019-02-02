import numpy as np
from camper.variables import PREDICT, PROBA
from sklearn.metrics import (
    roc_auc_score, accuracy_score, make_scorer
)


ROC_AUC = ['ROC_AUC', 'roc_auc']
ACCURACY = ['ACCURACY', 'accuracy', 'acc']
RMSLE = ['RMSLE', 'rmsle']


def rmsle(y, y_):
    """1 - rmsle, the greater the better"""
    assert len(y) == len(y_)
    return 1 - (np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_), 2))))


class Metrics:

    def __init__(self, metric, eval_type=None, greater_is_better=True):
        self.score = 0
        self.name = metric
        self.greater_is_better = greater_is_better
        if metric in ROC_AUC:
            self.metric = roc_auc_score
            self.eval_type = PROBA
        elif metric in ACCURACY:
            self.metric = accuracy_score
            self.eval_type = PREDICT
        elif metric in RMSLE:
            self.greater_is_better = False
            self.metric = rmsle
            self.eval_type = PREDICT
        else:
            print('Using a custom metric, {}'.format(metric))
            self.metric = metric
            self.eval_type = eval_type or PREDICT
            try:
                self.name = metric.__name__
            except AttributeError:
                raise ValueError('custom metric {} must be a function'.format(metric))

        self.need_proba = True if self.eval_type == PROBA else False

    def __call__(self, *args, **kwargs):
        self.score = self.metric(*args, **kwargs)
        return self.score

    def scorer(self):
        if self.eval_type == PROBA:
            return self.proba_scorer
        else:
            return self

    def make_scorer(self):
        score_func = self.proba_scorer if self.eval_type == PROBA else self
        return make_scorer(score_func, self.greater_is_better, self.need_proba)

    def proba_scorer(self, gt, pred):
        """Probability scorer on probability of a single class"""
        return self(gt, pred[:, 1])

    @property
    def __name__(self):
        return self.name

    def __str__(self):
        return "Metric({})".format(self.name)

    def __repr__(self):
        return self.__str__()
