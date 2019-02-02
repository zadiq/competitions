import os
import time
import json
from contextlib import contextmanager
from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf


data_root_dir = '/media/zadiq/ZHD/datasets/home_credit/'
file_map = {
    'app_train': 'application_train.csv',
    'app_test': 'application_test.csv',
    'bureau': 'bureau.csv',
    'bureau_bal': 'bureau_balance.csv',
    'cc_bal': 'credit_card_balance.csv',
    'ins_pay': 'installments_payments.csv',
    'pc_bal': 'POS_CASH_balance.csv',
    'pre_app': 'previous_application.csv',
    'col_desc': 'HomeCredit_columns_description.csv',

    'board': 'model_board.json',
    'board_train_prob': 'curated/{v}/board_train_probabilities.csv',
    'board_test_prob': 'curated/{v}/board_test_probabilities.csv',
    'board_models': 'curated/{v}/board_models',
    'board_models_ranks': 'curated/{v}/board_models/ranks.json',

    'submission': 'submissions/',
    'sub_meta': 'submissions/meta.json',

    # --- curated data and models ---
    'curated': 'curated/{v}',
    'all_data': 'curated/{v}/all.csv',
    'all_gen_data': 'curated/{v}/all_genetics.csv',
    'genes': 'curated/{v}/genes.json',
    'genes_sample': 'curated/{v}/genes_sample.json',
    'genes_bin_dir': 'curated/{v}/bins',
    'gen_train': 'curated/{v}/train_genetics.csv',
    'gen_test': 'curated/{v}/test_genetics.csv',
    'org_train': 'curated/{v}/original_train.csv',
    'org_test': 'curated/{v}/original_test.csv',
    'models': 'curated/{v}/models/',
    'ga_rank': 'curated/{v}/ga_rank.json',
    'sample_ga_rank': 'curated/{v}/ga_rank.json',
    'ga_ckpt': 'curated/{v}/checkpoints',
}
TRAIN_MODE = 'train_mode'
EVAL_MODE = 'eval_mode'
TEST_MODE = 'test_mode'
RAW_GENE = 'raw_gene'
ART_GENE = 'artificial_gene'
PLUS = '+'
SUBTRACT = '-'
DIVIDE = '/'
MULTIPLY = '*'


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def get_file(name, v=''):
    path = os.path.join(data_root_dir, file_map[name])
    path = path.format(v=v) if v else path
    return path


def folder_date():
    return datetime.now().strftime('%m-%d_%H-%M-%S')


def display_importances(feature_importance_df):
    cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                  ascending=False)[
           :40].index
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def auc_roc(y_true, y_pred):
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def calc_steps(batch_size, size):
    bs = batch_size
    return (size // bs) + (size % bs > 0)


def safe_load_json(path):
    try:
        with open(path) as fp:
            return json.load(fp)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def dump_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=2)


sample_ga_model = {
    'pop_size': 5,
    'chromosome_size': 7,
    'workers': 6,
    'lazy_size': 30,
    'mate_method': 0,
    'mate_numbers': 2,
    'pop_reverse_sort': True,
    'mutate_chance': .8,
    'log_threshold': .7,
    'mutate_scale': 2,
}

sample_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'nthread': 4,
    'learning_rate': 0.02,  # 02,
    'num_leaves': 10,
    'colsample_bytree': 0.9497036,
    'subsample': 0.8715623,
    'subsample_freq': 1,
    'max_depth': 2,
    'reg_alpha': 0.041545473,
    'reg_lambda': 0.0735294,
    'min_split_gain': 0.0222415,
    'min_child_weight': 60,  # 39.3259775,
    'seed': 0,
    'verbose': -1,
    'metric': 'auc',
    'device_type': 'cpu',
    'gpu_device_id': 0,
    'min_data': 1
}
