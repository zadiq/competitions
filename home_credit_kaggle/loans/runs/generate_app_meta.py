import pandas as pd
import json
from loans.utils import get_file
from sklearn.utils import class_weight

re_meta_data = {}

train_data = pd.read_csv(get_file('re_app_train'), index_col='SK_ID_CURR')
re_meta_data['train_size'] = int(train_data.shape[0])
re_meta_data['col_size'] = int(train_data.shape[1]) - 1
classes = pd.unique(train_data['TARGET'])
weights = class_weight.compute_class_weight('balanced', classes, train_data['TARGET'].values)
re_meta_data['class_weights'] = dict(zip(map(int, classes), weights))
del train_data

eval_data = pd.read_csv(get_file('re_app_eval'), index_col='SK_ID_CURR')
re_meta_data['eval_size'] = int(eval_data.shape[0])
del eval_data

test_data = pd.read_csv(get_file('re_app_test'), index_col='SK_ID_CURR')
re_meta_data['test_size'] = int(test_data.shape[0])
del test_data

with open(get_file('re_meta'), 'w') as meta_fp:
    json.dump(re_meta_data, meta_fp)
