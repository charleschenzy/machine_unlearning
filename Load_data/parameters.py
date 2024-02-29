import argparse

def hyperparam():
    parser = argparse.ArgumentParser(
        description='Extract a percentage of the data with the same distribution as the original dataset')
    parser.add_argument('--percentage', type=float, default=0.1, help='the percentage of data to extract')

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--batch_size', type=int, default=16384, help='batch size')

    parser.add_argument('--drop_last', type=bool, default=False, help='drop last')

    parser.add_argument('--device', type=int, default=1, help='device of cuda')

    parser.add_argument('--lr', type=float, default=0.1, help='lr of SGD')

    parser.add_argument('--T', type=int, default=50, help='number of epochs')

    parser.add_argument('--L2', type=float, default=0.005, help='L2_norm')

    parser.add_argument('--k', type=int, default=10, help='the epochs of unlearned information need to be stored')

    parser.add_argument('--m',type=int, default=3,help='the hyperparameter m for LBFGS')

    args = parser.parse_args()

    return args

args = hyperparam()