import matplotlib.pyplot as plt
import os
import random
import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm_notebook
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize as sk_res
from skimage.filters import rank, sobel
from skimage.morphology import disk, label, erosion, remove_small_objects
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage import io


class SaltDataset:

    def __init__(self, **params):
        
        self.validation_split = params['validation_split']
        self.database_dir = params['database_dir']
        self.img_shape = params['img_shape']
        self.train_img_shape = params['train_img_shape']
        self.seed = params['seed']
        self.x_gen_params = params.get('extra_gen_params', {})
        self.x_flow_params = params.get('extra_flow_params', {})
        load_test_gen = params.get('load_test_gen', False)
                
        self.img_path = os.path.join(self.database_dir, 'train', 'images')
        self.mask_path = os.path.join(self.database_dir, 'train', 'masks')
        self.aug_dir = self.get_aug_dir(params.get('augmented_folder'))

        self.image_samples = self.get_samples()
        self.mask_samples = self.get_samples('masks')

        self.gen_params = dict(
            fill_mode='constant', vertical_flip=True,
            horizontal_flip=True, rescale=1 / 255,
            validation_split=0.2
        )
        self.gen_params.update(self.x_gen_params)
        self.flow_params = dict(
            target_size=(self.train_img_shape[0], self.train_img_shape[0]),
            color_mode='grayscale', class_mode=None,
            seed=self.seed, save_to_dir=self.aug_dir,
            save_format='jpeg'
        )
        raw_flow_params = self.flow_params.copy()
        raw_flow_params['target_size'] = self.img_shape[:2]
        self.flow_params.update(self.x_flow_params)

        self.img_gen = ImageDataGenerator(**self.gen_params)
        self.mask_gen = ImageDataGenerator(**self.gen_params)
        raw_img_gen = ImageDataGenerator(rescale=1/255)
        raw_mask_gen = ImageDataGenerator(rescale=1/255)

        # fit generators
        self.img_gen.fit(self.image_samples, augment=True, seed=self.seed)
        self.mask_gen.fit(self.mask_samples, augment=True, seed=self.seed)

        # instantiate generators with directories
        train_img_gen = self.img_gen.flow_from_directory(
            self.img_path, save_prefix='img',
            subset='training', **self.flow_params
        )
        val_img_gen = self.img_gen.flow_from_directory(
            self.img_path, save_prefix='img',
            subset='validation', **self.flow_params
        )
        raw_img_gen = raw_img_gen.flow_from_directory(
            self.img_path, **raw_flow_params
        )

        train_mask_gen = self.mask_gen.flow_from_directory(
            self.mask_path, save_prefix='mask',
            subset='training', **self.flow_params
        )
        val_mask_gen = self.mask_gen.flow_from_directory(
            self.mask_path, save_prefix='mask',
            subset='validation', **self.flow_params
        )
        raw_mask_gen = raw_mask_gen.flow_from_directory(
            self.mask_path, **raw_flow_params
        )

        self.train_gen = zip(train_img_gen, train_mask_gen)
        self.val_gen = zip(val_img_gen, val_mask_gen)
        self.raw_gen = zip(raw_img_gen, raw_mask_gen)
        self.test_df = None
        self.train_rle_df = pd.read_csv(os.path.join(self.database_dir, 'train.csv'), index_col='id')
        self.test_gen = None
        self.test_batches = None
        if load_test_gen:
            self.test_gen = self.get_test_gen()

    @staticmethod
    def resize_img(img, shape):
        return sk_res(img, shape, mode='constant', preserve_range=True)

    @staticmethod
    def display_batch(batch, n=5):
        n = min(batch[0].shape[0], n)
        fig, axis = plt.subplots(n, 2, figsize=(15, 3 * n))
        for i, (img, mask) in enumerate(zip(*batch)):
            if i == n:
                break
            img_ax = axis[i, 0]
            img_ax.imshow(img.squeeze(), cmap="Greys")
            img_ax.set_title("Image")
            mask_ax = axis[i, 1]
            mask_ax.imshow(mask.squeeze(), cmap="Greys")
            mask_ax.set_title("Mask")

    @staticmethod
    def display_img(img):
        plt.imshow(img.squeeze(), cmap='Greys')
        plt.show()

    def get_samples(self, which='images', num=20):
        data_path = os.path.join(self.database_dir, 'train', which, which)
        images = os.listdir(data_path)
        images = [cv2.imread(os.path.join(data_path, img), 0).reshape(self.img_shape)
                  for img in random.sample(images, num)]
        return np.array(images) / 255

    def get_aug_dir(self, name):
        if not name:
            return
        path = os.path.join(self.database_dir, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    @property
    def get_test_df(self):
        if self.test_df is not None:
            return self.test_df

        test_dir = os.path.join(self.database_dir, 'test', 'images', 'images')
        images = os.listdir(test_dir)
        images_id = []
        images_path = []
        # img_array = []
        for img in tqdm_notebook(images, desc="test_df"):
            images_id.append(img.rstrip('.png'))
            path = os.path.join(test_dir, img)
            images_path.append(path)
            # img = cv2.imread(path, 0).reshape(self.img_shape) / 255
            # img = self.resize_img(img, self.train_img_shape[:2])
            # img_array.append(img)

        self.test_df = pd.DataFrame(dict(
            id=images_id, path=images_path,
        )).set_index('id')

        return self.test_df

    def get_test_gen(self, batch_size=32):
        # if self.test_gen:
        #     return self.test_gen

        _ = self.get_test_df

        self.test_batches = np.array_split(self.test_df, self.test_df.shape[0] // batch_size)

        def parse_row(row):
            img = cv2.imread(row.path, 0).reshape(self.img_shape) / 255
            img = self.resize_img(img, self.train_img_shape[:2])
            return img

        def test_gen():
            _ = np.concatenate
            for b in tqdm_notebook(self.test_batches):
                yield _(b.apply(parse_row, axis=1).values).reshape(-1, *self.train_img_shape)

        return test_gen()

    def make_submission(self, model, folder, which, threshold=.1, n=False):
        for i, batch in enumerate(self.get_test_gen()):
            rle_mask = [self.encode_rle(pred, threshold) for pred in model.predict(batch)]
            self.test_batches[i]['rle_mask'] = rle_mask
            if n and i >= n:
                break
        submission = pd.concat(self.test_batches, sort=True)
        _ = os.path.join
        sub_dir = _(self.database_dir, 'submissions')
        os.makedirs(sub_dir, exist_ok=True)
        path = _(sub_dir, '{}-{}-{}-submission.csv'.format(folder, which, threshold))
        submission.drop('path', axis=1).to_csv(path)
        return submission

    def visualize_submission(self, submission, n=5):

        def parse_imgs(path):
            return cv2.imread(path, 0).reshape(self.img_shape) / 255

        samples = submission.sample(n)
        imgs = samples.path.apply(parse_imgs).values
        masks = samples.rle_mask.apply(self.decode_rle).values
        self.display_batch((imgs, masks), n)

    def decode_rle(self, encoded):
        shape = self.img_shape[:2]
        img = np.zeros(shape).ravel()
        if not pd.isna(encoded) or encoded != "":
            s = encoded.split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
        return img.reshape(shape).T

    def encode_rle(self, mask: np.array, threshold=0.5, order='F', _format=True):
        """
        mask: np.array: 1 - mask, 0 - background
        Returns
        -------
        run-length string of pairs of (start, length)
        """
        mask = mask.squeeze()
        mask = (mask > threshold).astype('int16')
        mask = self.resize_img(mask, self.img_shape[:2])
        pixels = mask.reshape(mask.shape[0] * mask.shape[1], order=order)
        runs = []
        r = 0
        pos = 1
        for c in pixels:
            if c == 0:
                if r != 0:
                    runs.append((pos, r))
                    pos += r
                    r = 0
                pos += 1
            else:
                r += 1

        if r != 0:
            runs.append((pos, r))
            pos += r
            r = 0

        if _format:
            z = ''

            for rr in runs:
                z += '{} {} '.format(rr[0], rr[1])
            return z[:-1]
        else:
            return runs


class SaltProcess:

    def __init__(self, **params):
        self.database_dir = params['database_dir']
        # self.img_shape = params['img_shape']
        self.sobel_threshold = params.get('sobel_threshold', .05)
        self.sobel_disk_size = params.get('sobel_threshold', 2)
        self.img_shape = params.get('img_shape', (101, 101))

        self.train_df = None
        self.test_df = None

        self.parse_order = ['edge_counts', 'mid_point', 'distance']
        self.parse_map = {
            'edge_counts': None,
            'mid_point': None,
            'distance': None,
        }

        def _(*paths, is_dir=True):
            co_path = os.path.join(*paths)
            if is_dir:
                os.makedirs(co_path, exist_ok=True)
            return co_path

        train_base = _(self.database_dir, 'train')
        self.train_paths = {
            'edges': _(train_base, 'edges', 'edges'),
            'sobels': _(train_base, 'sobels', 'sobels'),
            'sobels_mask_v1': _(train_base, 'sobels_mask_v1', 'sobels_mask_v1'),
            'images': _(train_base, 'images', 'images'),
            'masks': _(train_base, 'masks', 'masks'),
            'combined_v1': _(train_base, 'combined_v1', 'combined_v1'),
            'meta': _(train_base, 'meta.csv', is_dir=False),
        }

        test_base = _(self.database_dir, 'test')
        self.test_paths = {
            'sobels': _(test_base, 'sobels', 'sobels'),
            'sobels_mask_v1': _(test_base, 'sobels_mask_v1', 'sobels_mask_v1'),
            'images': _(test_base, 'images', 'images'),
            'masks': _(test_base, 'masks', 'masks'),
            'combined_v1': _(test_base, 'combined_v1', 'combined_v1'),
            'meta': _(test_base, 'meta.csv', is_dir=False),
        }

        self.train_df = self.generate_df('train')
        self.test_df = self.generate_df('test')

    def generate_df(self, which, force=False):
        """
        :param which: train | test
        :param force: force regeneration
        :return:
        """
        which_map = {
            'train': (self.train_df, 'train'),
            'test': (self.test_df, 'test')
        }
        df, name = which_map[which]

        if df is not None and not force:
            return df

        is_train = which == 'train'
        mask_dir = None

        if is_train:
            img_dir = self.train_paths['images']
            mask_dir = self.train_paths['masks']
        else:
            img_dir = self.test_paths['images']

        images = os.listdir(img_dir)
        images_id = []
        images_path = []
        masks_path = []
        for img in tqdm_notebook(images, desc="test_df"):
            images_id.append(img.rstrip('.png'))
            images_path.append(os.path.join(img_dir, img))
            if is_train:
                masks_path.append(os.path.join(mask_dir, img))

        df = pd.DataFrame(dict(
            id=images_id, path=images_path,
        )).set_index('id', drop=False)

        if is_train:
            df['mask_path'] = masks_path
            self.train_df = df
        else:
            self.test_df = df

        return df

    @staticmethod
    def coord_distance(point_a, point_b):
        diff = np.array(point_b) - np.array(point_a)
        distance = np.sqrt((diff ** 2).sum()).round()
        return distance

    @staticmethod
    def arr_to_tuple(array):
        return tuple(np.array(array).astype('int8'))

    def get_mid_point(self, arr):
        """Get line mid point"""
        x, y = np.where(arr == 1)
        if np.any(x):
            return x[x.shape[0] // 2], y[y.shape[0] // 2]
        return self.img_shape[0]//2, self.img_shape[0]//2

    def get_edge_details(self, mask_edge):
        """
        :param mask_edge:
        :return:
            edge_counts: number of edge detected
            mid_point: mid point between two edges or centroid of edge if one,
            distance: distance between two edges, None if one edge
        """
        labels, counts = label(mask_edge, return_num=True)

        if counts <= 1:
            mid_point = self.get_mid_point(mask_edge)
            return counts, mid_point, None

        centroids = []
        for r in regionprops(labels):
            centroids.append(r.centroid)

        point_1 = np.array(centroids[0])
        point_2 = np.array(centroids[1])

        distance = self.coord_distance(point_1, point_2)
        diff = point_2 - point_1
        mid_point = point_1 + (.5 * diff)

        return counts, self.arr_to_tuple(mid_point), distance

    def parse_df(self, row, which='train'):
        _ = os.path.join
        parsed = self.parse_map.copy()
        img_name = "{}.png".format(row.id)
        img = (cv2.imread(row.path, 0) / 255).squeeze()
        is_train = which == 'train'

        paths = self.train_paths if is_train else self.test_paths

        if is_train:
            mask = (cv2.imread(row.mask_path, 0) / 255).squeeze()
            mask_edge = rank.entropy(mask, disk(1)).round()
            parsed['edge_counts'], parsed['mid_point'], parsed['distance'] = self.get_edge_details(mask_edge)
            # io.imsave(_(paths['edges'], img_name), mask_edge)

        sobel_img = sobel(img)
        io.imsave(_(paths['sobels'], img_name), sobel_img)

        sobel_bin = sobel_img < self.sobel_threshold
        sobel_er = erosion(sobel_bin, disk(self.sobel_disk_size))
        sobel_mask = remove_small_objects(ndi.binary_opening(sobel_er))
        io.imsave(_(paths['sobels_mask_v1'], img_name), sobel_mask.astype('int8') * 255)

        combined_image = np.dstack((
            img, sobel_img, sobel_mask
        ))
        io.imsave(_(paths['combined_v1'], img_name), combined_image)

        return [parsed[k] for k in self.parse_order]

    def process(self, which='train'):
        df = self.train_df if which == 'train' else self.test_df
        paths = self.train_paths if which == 'train' else self.test_paths
        p = self.parse_order
        df[p[0]], df[p[1]], df[p[2]] = zip(*df.apply(self.parse_df, args=(which, ), axis=1))
        df.to_csv(paths['meta'])

    @property
    def train_gen(self):
        self.generate_df('train')
        yield from self.train_df

    @property
    def test_gen(self):
        self.generate_df('test')
        yield from self.test_df
