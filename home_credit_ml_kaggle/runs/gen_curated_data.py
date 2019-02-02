import sys
sys.path.append('/home/zoguntim/dev/home_credit_ml')
from credit.data import CurateData
from credit.utils import timer


if __name__ == "__main__":
    with timer("Generate curated data"):
        data = CurateData()
        data.gen_data()
    print('Finished!')
