import sys
sys.path.append('/home/zadiq/dev/home_credit_ml')

from credit.ga import CreditGA, TrainConfig, SAMPLE, MULTI, OFF


if __name__ == '__main__':

    if SAMPLE:
        import pandas as pd
        import numpy as np
        sample_size = 50
        sample = pd.DataFrame({
            'SK_ID_CURR': [x for x in range(sample_size)],
            'TARGET': [np.random.randint(0, 2) for _ in range(sample_size // 2)] + [np.nan for _ in range(sample_size // 2)],
            'A': [np.random.randint(0, 100) for _ in range(sample_size)],
            'B': [np.random.randint(0, 100) for _ in range(sample_size)],
            'C': [np.random.randint(0, 100) for _ in range(sample_size)],
            'D': [np.random.randint(0, 100) for _ in range(sample_size)],
            'E': [np.random.randint(0, 100) for _ in range(sample_size)],
            'F': [np.random.randint(0, 100) for _ in range(sample_size)],
        })
        model = CreditGA(sample_df=sample)
    else:
        tc = TrainConfig()
        if OFF:
            tc.ga_model['workers'] = 2
            tc.ga_model['pop_size'] = 20
            tc.ga_model['mate_numbers'] = 20
            tc.ga_model['mutate_scale'] = 10
            tc.ga_model['chromosome_size'] = 500
        if MULTI:
            tc.gpu_devices = [3]
            tc.ga_model['chromosome_size'] = 1000
            tc.ga_model['lazy_size'] = 50
            tc.ga_model['workers'] = 4
            tc.ga_model['pop_size'] = 20
            tc.ga_model['mate_numbers'] = 20
            tc.ga_model['mutate_scale'] = 20
            tc.num_folds = 5
            tc.stratified = True
        model = CreditGA(tc=tc)
        model.evolve()
