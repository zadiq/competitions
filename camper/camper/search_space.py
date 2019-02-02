from skopt.space import Real, Integer, Categorical

# Classifiers #

ET_CLF = {  # ExtraTreesClassifier
    'n_estimators': Integer(3000, 10000),
    'max_features': Categorical(['auto', 'sqrt', 'log2', None]),
    'max_depth': Integer(5, 100),
    'min_samples_split': Real(0.01, 1),
    'min_samples_leaf': Real(0.01, .5),  # no
    # 'min_impurity_decrease': Real(0, 1),  # maybe
    'max_leaf_nodes': Categorical([10, 20, 30, None]),
    'class_weight': Categorical(['balanced', 'balanced_subsample', None]),
}

RF_CLF = ET_CLF.copy()  # RandomForestClassifier

GB_CLF = {  # GradientBoostingClassifier
    'learning_rate': Real(0.001, 1),
    'n_estimators': Integer(3000, 10000),
    'max_depth': Integer(5, 50),
    'min_samples_split': Real(0.01, 1),
    'min_samples_leaf': Real(0.01, .5),
    'min_weight_fraction_leaf': Real(0, .5),
    'max_features': Categorical(['auto', 'sqrt', 'log2', None]),
    'max_leaf_nodes': Categorical([10, 20, 30, None]),
    # 'min_impurity_decrease': Real(0, 1),
}

AB_CLF = {  # AdaBoostClassifier
    'base_estimator': Categorical([None]),
    'n_estimators': Integer(3000, 10000),
}

LGBM_CLF = {  # LGBMClassifier
    'boosting_type': Categorical(['gbdt', 'dart', 'rf']),
    'n_estimators': Integer(3000, 10000),
    'num_leaves': Integer(20, 35),
    'max_depth': Integer(5, 50),
    'is_unbalance': Categorical([True, False]),
    'learning_rate': Real(0.01, 0.5),
    'min_child_weight': Real(0.1, 100),
    'min_split_gain': Real(0, 0.3),
    'min_child_samples': Integer(10, 50),
    'subsample': Real(.01, .9),
    'feature_fraction': Real(.01, .9),
    'subsample_freq': Integer(1, 20),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0, 1),
}

KN_CLF = {  # KNeighborsClassifier
    'n_neighbors': Integer(5, 200),
    'weights': Categorical(['uniform', 'distance']),
    'leaf_size': Integer(10, 50),
    'p': Integer(1, 3),
}

SVC_CLF = {  # SVC
    'C': Real(0.001, 5),
    'gamma': Real(0.0001, 1),
    'tol': Real(1e-5, 1e-1),
    'probability': Categorical([True]),
}

GP_CLF = {  # GaussianProcessClassifier
    'max_iter_predict': Integer(200, 1000)
}

GNB_CLF = {  # GaussianNB
    'priors': Categorical([None])
}

QDA_CLF = {  # QuadraticDiscriminantAnalysis
    'reg_param': Real(0, 1),
    'tol': Real(1e-5, 1e-1),
}


# Regressors #


ET_REG = {  # ExtraTreesRegressor
    'n_estimators': Integer(3000, 10000),
    'max_features': Categorical(['auto', 'sqrt', 'log2', None]),
    'max_depth': Integer(5, 100),
    'min_samples_split': Real(0.01, 1),
    'min_samples_leaf': Real(0.01, .5),  # no
    # 'min_impurity_decrease': Real(0, 1),  # maybe
    'max_leaf_nodes': Categorical([10, 20, 30, None]),
    'class_weight': Categorical(['balanced', 'balanced_subsample', None]),
    'oob_score': Categorical([True, False]),
}

RT_REG = ET_REG.copy()  # RandomForestRegressor

GB_REG = {  # GradientBoostingRegressor
    'loss': Categorical(['ls', 'lad', 'huber', 'quantile']),
    'learning_rate': Real(0.001, 1),
    'n_estimators': Integer(3000, 10000),
    'max_depth': Integer(5, 50),
    'min_samples_split': Real(0.01, 1),
    'min_samples_leaf': Real(0.01, .5),
    'min_weight_fraction_leaf': Real(0, .5),
    'max_features': Categorical(['auto', 'sqrt', 'log2', None]),
    'max_leaf_nodes': Categorical([10, 20, 30, None]),
    'alpha': Real(0.10, 0.99),
}

AB_REG = {  # AdaBoostRegressor
    'n_estimators': Integer(3000, 10000),
    'learning_rate': Real(0, 1),
    'loss': Categorical(['linear', 'square', 'exponential']),
}

LGBM_REG = {  # LGBMRegressor
    'boosting_type': Categorical(['gbdt', 'dart', 'rf']),
    'n_estimators': Integer(3000, 10000),
    'num_leaves': Integer(20, 35),
    'max_depth': Integer(5, 50),
    'is_unbalance': Categorical([True, False]),
    'learning_rate': Real(0.01, 0.5),
    'min_child_weight': Real(0.1, 100),
    'min_split_gain': Real(0, 0.3),
    'min_child_samples': Integer(10, 50),
    'subsample': Real(.01, .9),
    'feature_fraction': Real(.01, .9),
    'subsample_freq': Integer(1, 20),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0, 1),
}

KN_REG = {  # KNeighborsRegressor
    'n_neighbors': Integer(5, 200),
    'weights': Categorical(['uniform', 'distance']),
    'leaf_size': Integer(10, 50),
    'p': Integer(1, 3),
}

RIDGE_REG = {  # RidgeRegressor
    'alpha': Real(0.01, 5),
    'tol': Real(1e-5, 1e-1),
}

LIN_REG = {  # LinearRegression
    'normalize': Categorical([True, False]),
}

THEIL_REG = {  # ThielSenRegression
    'max_subpopulation': Integer(1e3, 1e10),
    'max_iter': Integer(1e2, 1e3),
    'tol': Real(1e-5, 1e-1),
}

ENET_REG = {  # ElasticNetCV
    'l1_ratio': Integer(0.01, .99),
    'eps': Real(1e-4, 1e-1),
    'alphas': Integer(50, 500),
    'normalize': Categorical([True, False]),
    'max_iter': Integer(700, 5000),
    'tol': Real(1e-5, 1e-1),
    'cv': Integer(10, 50),
}
