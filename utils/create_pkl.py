import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from params import real_images, fake_images, pkl_dir
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', help='P where PxP is the dimension of the patch', type=int, default=128)

    args = parser.parse_args()
    P = args.patch_size

    # extract real images:

    img_fmt = ('*.jpeg', '*.jpg', '*.png', '*.gif')  # image formats
    real_image_paths_list = []
    for dataset_dict in real_images:
        for dataset_dir in dataset_dict.keys():
            # print(dataset_dir)
            for fmt in img_fmt:
                real_image_paths_list.extend(sorted(glob(os.path.join(dataset_dict[dataset_dir], fmt))))

    real_df = pd.DataFrame(real_image_paths_list, columns=['image'])
    real_df['label'] = 0
    # shuffle the dataset
    real_df = real_df.sample(frac=1)

    # extract fake images:
    fake_image_paths_list = []
    for dataset_dict in fake_images:
        for dataset_dir in dataset_dict.keys():
            # print(dataset_dir)
            for fmt in img_fmt:
                fake_image_paths_list.extend(sorted(glob(os.path.join(dataset_dict[dataset_dir], fmt))))

    synth_df = pd.DataFrame(fake_image_paths_list, columns=['image'])
    synth_df['label'] = 1
    # shuffle the dataset
    synth_df = synth_df.sample(frac=1)

    df = pd.concat([synth_df, real_df]).reset_index().drop('index', axis=1)

    # balance the number of images per class
    min_len = np.min((len(df[df['label'] == 0]), len(df[df['label'] == 1])))

    df = df.groupby('label').apply(lambda x: x[:min_len])
    df.index = np.arange(len(df))

    print('maintain only readable images')
    out_df = df[df['image'].map(lambda x: Image.open(x).size != () and Image.open(x).size[0] >= P and
                                                  Image.open(x).size[1] >= P and Image.open(x).mode.startswith('RGB'))]

    os.makedirs(pkl_dir, mode=0o775, exist_ok=True)
    out_df.to_pickle(os.path.join(pkl_dir, 'all_P-{}.pkl'.format(P)))

    return 0


if __name__ == '__main__':
    main()
