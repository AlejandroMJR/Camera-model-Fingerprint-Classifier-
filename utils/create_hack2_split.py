import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from params import pkl_dir

np.random.seed(21)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='test percentage over the total amount of data', type=float, default=0.8)
    parser.add_argument('--val', help='val percentage over the amount of train data', type=float, default=0.5)
    parser.add_argument('--patch_size', help='P where PxP is the dimension of the patch', type=int, default=128)

    args = parser.parse_args()
    test_frac = args.test
    val_frac = args.val
    P = args.patch_size

    # read dataframe
    df = pd.read_pickle(os.path.join(pkl_dir, 'all_P-{}.pkl'.format(P)))

    # We want all the patches of an image either in train, val or test set
    image_groups = [x for _, x in df.groupby('image')]

    # Split
    train_val_groups, test_groups = train_test_split(image_groups, test_size=test_frac)
    train_groups, val_groups = train_test_split(train_val_groups, test_size=val_frac)

    # Re-create dataframes
    train_df = pd.concat(train_groups)
    val_df = pd.concat(val_groups)
    test_df = pd.concat(test_groups)

    # Save pickles
    train_df.to_pickle(os.path.join(pkl_dir, 'train_P-{}.pkl'.format(P)))
    val_df.to_pickle(os.path.join(pkl_dir, 'val_P-{}.pkl'.format(P)))
    test_df.to_pickle(os.path.join(pkl_dir, 'test_P-{}.pkl'.format(P)))

    return 0


if __name__ == '__main__':
    main()