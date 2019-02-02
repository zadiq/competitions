import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from credit.utils import (
    get_file,
)
from datetime import datetime


class TrainConfig:
    def __init__(self):
        self.version = 'v0'
        self.num_folds = 10
        self.stratified = False
        self.seed = 290
        self.use_class_weights = True
        self.board_model = {
            'which': 0,
            'layers': [100, 150, 200, 200, 150, 60],
            'dropouts': [],
            'layer_params': {'activation': 'relu'},
            'use_multi_gpu': True,
            'gpu_counts': 4,
            'optimizer': 'adam',
            'optimizer_params': {},
            'batch_size': 100,
            'epochs': 1000,
            'model_folder': None,
            'test_size': .30,
        }
        self.params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60,  # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
            'device_type': 'cpu',
            'gpu_device_id': 0,
        }

    @classmethod
    def from_json(cls, path):
        obj = cls()
        with open(path) as fp:
            config = json.load(fp)
            obj.__dict__.update(config)
            return obj

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def load_train_config(tc):
    if type(tc) == str:
        return TrainConfig.from_json(tc)
    return tc


def kfold_lightgbm(tc=TrainConfig(), manual=False, test_df=None, train_df=None):

    board = {}

    tc = load_train_config(tc)
    if not manual:
        train_df = pd.read_csv(get_file('org_train', v=tc.version), index_col='index')
        test_df = pd.read_csv(get_file('org_test', v=tc.version), index_col='index')
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    os.makedirs(get_file('models', v=tc.version), exist_ok=True)
    model_dir = datetime.now().strftime('%m-%d_%H-%M-%S')
    model_path = os.path.join(get_file('models', v=tc.version), model_dir)
    os.makedirs(model_path, exist_ok=True)
    sub_file = os.path.join(model_path, 'submission.csv')
    feat_imp_file = os.path.join(model_path, 'feature_importance.csv')
    model_txt = os.path.join(model_path, 'model-{score:.2f}.txt')
    tc_path = os.path.join(model_path, 'train_config.json')

    with open(tc_path, 'w') as tcp:
        json.dump(json.loads(tc.to_json()), tcp, indent=4)

    print('Model path: {}'.format(model_path))

    # Cross validation model
    if tc.stratified:
        folds = StratifiedKFold(n_splits=tc.num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=tc.num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    clf = None

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        d_train = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                              label=train_df['TARGET'].iloc[train_idx],
                              free_raw_data=False, silent=True)
        d_valid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                              label=train_df['TARGET'].iloc[valid_idx],
                              free_raw_data=False, silent=True)

        clf = lgb.train(
            params=tc.params,
            train_set=d_train,
            num_boost_round=10000,
            valid_sets=[d_train, d_valid],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        oof_preds[valid_idx] = clf.predict(d_valid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(d_valid.label, oof_preds[valid_idx])))

    # print('Saving model')
    score = roc_auc_score(train_df['TARGET'], oof_preds)
    mt = model_txt.format(score=score)
    board[mt] = score
    clf.save_model(mt)

    print('Full AUC score %.6f' % score)

    try:
        with open(get_file('board')) as fp:
            current_board = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        current_board = {}

    with open(get_file('board'), 'w') as fp:
        current_board.update(board)
        json.dump(current_board, fp)

    # Write submission file and plot feature importance
    sub_df = test_df[['SK_ID_CURR']].copy()
    sub_df['TARGET'] = sub_preds
    sub_df[['SK_ID_CURR', 'TARGET']].to_csv(sub_file, index=False)
    feature_importance_df.to_csv(feat_imp_file, index=False)
    return feature_importance_df, score, model_path


