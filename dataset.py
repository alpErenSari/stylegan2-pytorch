from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class MultiResolutionDatasetRandomCrop(Dataset):
    def __init__(self, path, transform, resolution=256, crop_res=(16,16)):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.crop_res = crop_res
        self.transform = transform
        # self.embedding = nn.Embedding((resolution - crop_res[0])**2, 128)

    def __len__(self):
        return self.length

    def random_crop_dataset(self, img):
        _, h, w = img.size()
        th, tw = self.crop_res
        number_patches_sqrt_h = h // th
        number_patches_sqrt_w = w // tw

        if h < th or w < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, number_patches_sqrt_h, size=(1,)).item()
        j = torch.randint(0, number_patches_sqrt_w, size=(1,)).item()

        position_label = torch.tensor(i * number_patches_sqrt_w + j).long()

        return img[:,i*th:(i+1)*th,j*tw:(j+1)*tw], position_label

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        img, position_label = self.random_crop_dataset(img)

        return img, position_label
