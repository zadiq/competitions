import sys
sys.path.append('/home/zoguntim/dev/home_credit_ml')
from credit.data import BoardProbabilities
from credit.utils import timer


if __name__ == "__main__":
    with timer("Generate board probabilities"):
        data = BoardProbabilities()
    print('Finished!')
