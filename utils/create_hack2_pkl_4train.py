import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from params import real_images, fake_images, pkl_dir
from glob import glob


def find_manipulation(row):
    found_manipulation = 'real' if row['label_gan_man_real'] == 2 else str(row['manipulation'].split('/')[0])
    return found_manipulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', help='P where PxP is the dimension of the patch', type=int, default=128)

    args = parser.parse_args()
    P = args.patch_size

    image_root = '/nas/public/exchange/semafor/hackaton2/cp3/hk2_cp3'
    img_fmt = ('*.jpeg', '*.jpg', '*.png', '*.gif')  # image formats

    # label delle immagini:
    labels_df = pd.read_csv('/nas/public/exchange/semafor/hackaton2/cp3/solution.csv', header=None)
    labels_real_fake = labels_df[0]
    labels_gan_man_real = labels_df[1]
    manipulations = labels_df[2]

    image_path_list = []
    for fmt in img_fmt:
        image_path_list.extend(sorted(glob(os.path.join(image_root, fmt))))

    df = pd.DataFrame(image_path_list, columns=['image'])

    df['label_real_fake'] = labels_real_fake.values
    df['label_gan_man_real'] = labels_gan_man_real.values
    df['manipulation'] = manipulations.values
    df['manipulation'] = df.apply(lambda row: find_manipulation(row), axis=1)

    # togli le manipolate.
    df = df[df['label_gan_man_real'] != 1]

    # assegna le label
    df['label'] = df['label_gan_man_real'].map(lambda x: 0 if x==2 else 1)

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
