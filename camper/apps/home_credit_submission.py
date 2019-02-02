import pandas as pd
import numpy as np
from camper.models import VotingClassifier
from sklearn.preprocessing import Imputer

which = 'test'
voter_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/voters_log/{}/voter.pkl'.format(which)
data_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/all.csv'
sub_path = '/media/zadiq/ZHD/datasets/home_credit/curated/v1/voters_log/{}/submission.csv'.format(which)


if __name__ == '__main__':

    vc = VotingClassifier.from_pkl(voter_path)
    dataset = pd.read_csv(data_path, dtype='float32')
    dataset = dataset[dataset['TARGET'].isnull()]
    features = [f for f in dataset.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    x = dataset[features]
    imp = Imputer(np.nan, copy=False)
    x = x.replace([np.inf, -np.inf], np.nan)
    x[x.columns] = imp.fit_transform(x)
    sub = pd.DataFrame()
    sub['SK_ID_CURR'] = dataset['SK_ID_CURR'].astype('int').values
    sub['TARGET'] = vc.predict_proba(x)[:, 1]

    sub.to_csv(sub_path, index=False)
