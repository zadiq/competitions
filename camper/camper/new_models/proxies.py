from camper.search_space import (
    ET_CLF, RF_CLF, GB_CLF, AB_CLF,
    LGBM_CLF, KN_CLF, SVC_CLF, GP_CLF,
    GNB_CLF, QDA_CLF
)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesRegressor, RandomForestRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import (
    RidgeCV, LinearRegression, TheilSenRegressor,
    ElasticNetCV, LassoLarsCV, BayesianRidge, SGDRegressor,
    HuberRegressor, LassoCV
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR


class EstimatorProxyBase:
    klass = None
    has_feature_imp = False
    estimator_type = None
    search_space = None

    def __init__(self):
        pass

    def __str__(self):
        return "EstimatorProxy({})".format(self.klass.__name__)

    def __repr__(self):
        return self.__str__()


###############
# CLASSIFIERS #
###############


class ClassifierProxyBase(EstimatorProxyBase):
    estimator_type = 'classifier'


class ExtraTreesClassifierProxy(ClassifierProxyBase):
    klass = ExtraTreesClassifier
    has_feature_imp = True
    search_space = ET_CLF


class RandomForestClassifierProxy(ClassifierProxyBase):
    klass = RandomForestClassifier
    has_feature_imp = True
    search_space = RF_CLF


class GradientBoostingClassifierProxy(ClassifierProxyBase):
    klass = GradientBoostingClassifier
    has_feature_imp = True
    search_space = GB_CLF


class AdaBoostClassifierProxy(ClassifierProxyBase):
    klass = AdaBoostClassifier
    has_feature_imp = True
    search_space = AB_CLF


class LGBMClassifierProxy(ClassifierProxyBase):
    klass = LGBMClassifier
    has_feature_imp = True
    search_space = LGBM_CLF


class KNeighborsClassifierProxy(ClassifierProxyBase):
    klass = KNeighborsClassifier
    search_space = KN_CLF


class GaussianProcessClassifierProxy(ClassifierProxyBase):
    klass = GaussianProcessClassifier
    search_space = GP_CLF


class GaussianNBClassifierProxy(ClassifierProxyBase):
    klass = GaussianNB
    search_space = GNB_CLF


class QDAClassifierProxy(ClassifierProxyBase):
    klass = QuadraticDiscriminantAnalysis
    search_space = QDA_CLF


class SVCProxy(ClassifierProxyBase):
    klass = SVC
    search_space = SVC_CLF


class DecisionTreeClassifierProxy(ClassifierProxyBase):
    # TODO implement search_space
    klass = DecisionTreeClassifier
    has_feature_imp = True


##############
# REGRESSORS #
##############


class RegressorProxyBase(EstimatorProxyBase):
    estimator_type = 'regressor'


class ExtraTreesRegressorProxy(RegressorProxyBase):
    klass = ExtraTreesRegressor
    has_feature_imp = True
    search_space = ET_CLF


class RandomForestRegressorProxy(RegressorProxyBase):
    klass = RandomForestRegressor
    has_feature_imp = True
    search_space = RF_CLF


class GradientBoostingRegressorProxy(RegressorProxyBase):
    klass = GradientBoostingRegressor
    has_feature_imp = True
    search_space = GB_CLF


class AdaBoostRegressorProxy(RegressorProxyBase):
    klass = AdaBoostRegressor
    has_feature_imp = True
    search_space = AB_CLF


class LGBMRegressorProxy(RegressorProxyBase):
    klass = LGBMRegressor
    has_feature_imp = True
    search_space = LGBM_CLF


class KNeighborsRegressorProxy(RegressorProxyBase):
    klass = KNeighborsRegressor
    search_space = KN_CLF


class RidgeRegressorProxy(RegressorProxyBase):
    klass = RidgeCV
    search_space = KN_CLF


class LinearRegressorProxy(RegressorProxyBase):
    klass = LinearRegression
    search_space = KN_CLF


class TheilSenRegressorProxy(RegressorProxyBase):
    klass = TheilSenRegressor
    search_space = KN_CLF


class ElasticNetCVRegressorProxy(RegressorProxyBase):
    klass = ElasticNetCV
    search_space = KN_CLF


class LassoLarsCVRegressorProxy(RegressorProxyBase):
    klass = LassoLarsCV
    search_space = KN_CLF


class BayesianRidgeRegressorProxy(RegressorProxyBase):
    klass = BayesianRidge
    search_space = KN_CLF


class SGDRegressorProxy(RegressorProxyBase):
    klass = SGDRegressor
    search_space = KN_CLF


class HuberRegressorProxy(RegressorProxyBase):
    klass = HuberRegressor
    search_space = KN_CLF


class LassoCVRegressorProxy(RegressorProxyBase):
    klass = LassoCV
    search_space = KN_CLF


class KernelRidgeRegressorProxy(RegressorProxyBase):
    klass = KernelRidge
    search_space = KN_CLF


class GaussianProcessRegressorProxy(RegressorProxyBase):
    klass = GaussianProcessRegressor
    search_space = KN_CLF


class SVRRegressorProxy(RegressorProxyBase):
    klass = SVR
    search_space = KN_CLF


PROXY_MAP = {
    # classifiers
    'et_clf': ExtraTreesClassifierProxy,
    'rf_clf': RandomForestClassifierProxy,
    'gb_clf': GradientBoostingClassifierProxy,
    'ab_clf': AdaBoostClassifierProxy,
    'lgbm_clf': LGBMClassifierProxy,
    'kn_clf': KNeighborsClassifierProxy,
    'gp_clf': GaussianProcessClassifierProxy,
    'gnb_clf': GaussianNBClassifierProxy,
    'qda_clf': QDAClassifierProxy,
    'svc_clf': SVCProxy,
    'dt_clf': DecisionTreeClassifierProxy,

    # regressors
    'et_reg': ExtraTreesRegressorProxy,
    'rf_reg': RandomForestRegressorProxy,
    'gb_reg': GradientBoostingRegressorProxy,
    'ab_reg': AdaBoostRegressorProxy,
    'lgbm_reg': LGBMRegressorProxy,
    'kn_reg': KNeighborsRegressorProxy,
    'lin_reg': LinearRegressorProxy,
    'theil_reg': TheilSenRegressorProxy,
    'enet_reg': ElasticNetCVRegressorProxy,
    'lasso_lars_reg': LassoLarsCVRegressorProxy,
    'bayes_reg': BayesianRidgeRegressorProxy,
    'sgd_reg': SGDRegressorProxy,
    'huber_reg': HuberRegressorProxy,
    'lasso_reg': LassoCVRegressorProxy,
    'ridge_reg': RidgeRegressorProxy,
    'k_ridge_reg': KernelRidgeRegressorProxy,
    'gaussian_reg': GaussianProcessRegressorProxy,
    'svr_reg': SVRRegressorProxy,
}
