import os
import numpy as np
import pandas as pd


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
    'cleaned_app_train': 'cleaned_application_train.csv',
    'valid_app_col': 'valid_app_columns.json',
    'app_unique_voc': 'application_unique_vocabulary.json',
    'model_dir': 'models/',
    'model_dir2': 'models_2',
    'model_linear_wide': 'model_linear_wide',
    'export_model_dir': 'models/exported/',

    # regrouped data
    're_app_train': 'regrouped/re_app_train.csv',
    're_app_eval': 'regrouped/re_app_eval.csv',
    're_app_test': 'regrouped/re_app_test.csv',
    're_meta': 'regrouped/re_meta.json',

    # combined data
    'c_pre_app': 'combined/prev_app.csv',
    'c_pre_app_meta': 'combined/prev_app.json',
    'c_bu': 'combined/bureau.csv',
    'c_bu_meta': 'combined/bureau.json',
    'c_cc_bal': 'combined/credit_card_balance.csv',
    'c_cc_bal_meta': 'combined/credit_card_balance.json',
    'c_pc_bal': 'combined/POS_CASH_balance.csv',
    'c_pc_bal_meta': 'combined/POS_CASH_balance.json',
    'c_ins_pay': 'combined/installments_payments.csv',
    'c_ins_pay_meta': 'combined/installments_payments.json',
    'c_app': 'combined/application.csv',
    'c_app_meta': 'combined/application.json',
    'merged': 'combined/merged.csv',
    'merged_test': 'combined/merged_test.csv',
    'merged_meta': 'combined/merged_meta_data.json',
    
    'model_train': 'combined/merged_model_train.csv',
    'model_eval': 'combined/merged_model_eval.csv',
    'model_test': 'combined/merged_model_test.csv',
}
NOT_PROVIDED = 'NOT_PROVIDED'
TRAIN_MODE = 'train_mode'
EVAL_MODE = 'eval_mode'
TEST_MODE = 'test_mode'


def get_file(name):
    return os.path.join(data_root_dir, file_map[name])


def one_hot_encode(df):
    oc = df.columns.tolist()
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    nc = [c for c in df.columns if c not in oc]
    return df, nc


def generate_meta(df, prefix):
    prefix += '_'
    meta = {
        'cols': [],
        'dtype': [],
    }
    for c in df.columns:
        if not c.startswith(prefix):
            raise ValueError('Column name {} missing prefix {}'.format(c, prefix))
        meta['dtype'].append(str(df[c].dtype))
        meta['cols'].append(c)

    df.columns = meta['cols']

    return meta


def get_missing_col(data):
    miss_column = data.isnull().sum()
    miss_column = miss_column[miss_column>0]
    return miss_column.sort_values(ascending=False)/data.shape[0] * 100


def calc_harmonic_mean(series, _round=0):
    """
    calc harmonic mean of data column
    """
    mean = series.count() / (1 / series).sum()
    return np.around(mean, _round)


def calc_col_mean_by_target(data, column, target, use_ar_mean=False, _round=0):
    """
    find the harmonic mean of column
    in respect to the target.
    """
    # get column data where it is not NaN and target is 0
    series_t0 = data[column][data[target] == 0]
    series_t0 = series_t0[series_t0.isna() == False]
    # get column data where it is not NaN and target is 1
    series_t1 = data[column][data[target] == 1]
    series_t1 = series_t1[series_t1.isna() == False]

    # check that total number of series == to column data not NaN count
    total_series_count = series_t0.count() + series_t1.count()
    assert total_series_count == data[column][data[column].isna() == False].count()

    if use_ar_mean:
        # use arithmetic mean
        ar_mean_0 = series_t0.mean()
        ar_mean_1 = series_t1.mean()
        return np.around(ar_mean_0, _round), np.around(ar_mean_1, _round)

    return calc_harmonic_mean(series_t0, _round), calc_harmonic_mean(series_t1, _round)


def nan_to_mean(x, col, mean_dict, target):
    return mean_dict[col][int(x[target])]


def replace_nan_with_mean(data, columns, target, mean_dict):
    for c in columns:
        mask = data[c].isna()
        is_nan = data[[c, target]][mask]
        data.loc[mask, c] = is_nan.apply(nan_to_mean, args=(c, mean_dict, target), axis=1)


categorical_col = [
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'FLAG_MOBIL',
    'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE',
    'FLAG_PHONE',
    'FLAG_EMAIL',
    'OCCUPATION_TYPE',
    'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY',
    'WEEKDAY_APPR_PROCESS_START',
    'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY',
    'ORGANIZATION_TYPE',
    'FONDKAPREMONT_MODE',  # CONSIDER
    'HOUSETYPE_MODE',  # CONSIDER
    'WALLSMATERIAL_MODE',  # CONSIDER
    'EMERGENCYSTATE_MODE',  # CONSIDER
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8',
    'FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12',
    'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15',
    'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17',
    'FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20',
    'FLAG_DOCUMENT_21',
]
