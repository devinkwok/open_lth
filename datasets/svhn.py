# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import sys
import torchvision

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset, base.NdarrayDataset):
    """The SVHN dataset."""

    @staticmethod
    def num_train_examples(): return 73257

    @staticmethod
    def num_test_examples(): return 26032

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_data(train):
        dataset = torchvision.datasets.SVHN(
                root=os.path.join(get_platform().dataset_root, 'svhn'),
                split='train' if train else 'test', download=True)
        return dataset.data, np.array(dataset.labels)

    @staticmethod
    def get_train_set(use_augmentation, train_split=None):
        augment = []
        data, targets = Dataset.get_data_split(True, train_split)
        return Dataset(data, targets, augment if use_augmentation else [])

    @staticmethod
    def get_test_set(test_split=None):
        data, targets = Dataset.get_data_split(False, test_split)
        return Dataset(data, targets)

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
            [torchvision.transforms.Normalize(mean=[0.438, 0.444, 0.473], std=[0.198, 0.201, 0.197])])

    def example_to_image(self, example):
        return Image.fromarray(np.transpose(example, (1, 2, 0)))


DataLoader = base.DataLoader
