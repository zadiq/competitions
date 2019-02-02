from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from lightgbm import LGBMClassifier

# MODEL BACKENDS
SCIKIT = 'SCIKIT'
LIGHT_GBM = 'LIGHT_GBM'

# PREDICTION TYPE
PREDICT = 'PREDICT'
PROBA = 'PROBABILITY'

CLF_WITH_FI = [
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    LGBMClassifier
]
