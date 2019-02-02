import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


base_dir = '/media/zadiq/ZHD/datasets/salt'
shape = (101, 101)
train = pd.read_csv(os.path.join(base_dir, 'train.csv'), index_col='id')


def decode(encoded):
    img = np.zeros(shape).ravel()
    if not pd.isna(encoded):
        s = encoded.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape).T


def decode_sample():
    sample = train.sample(1)
    plt.imshow(decode(sample['rle_mask'].values[0]))
    plt.title(sample.index[0], )
    plt.show()


def rle_encode(img):
    """
    img: np.array: 1 - mask, 0 - background
    Returns
    -------
    run-length string of pairs of (start, length)
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle if rle else float('nan')


if __name__ == '__main__':
    decode_sample()
