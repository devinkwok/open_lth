# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset, base.NdarrayDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def _get_data(train):
        dataset = torchvision.datasets.MNIST(train=train, root=os.path.join(
            get_platform().dataset_root, 'mnist'), download=get_platform().download_data)
        return dataset.data, np.array(dataset.targets)

    @staticmethod
    def get_train_set(use_augmentation, train_split=None):
        # No augmentation for MNIST.
        data, targets = Dataset.get_data_split(True, train_split)
        return Dataset(data, targets)

    @staticmethod
    def get_test_set(test_split=None):
        data, targets = Dataset.get_data_split(False, test_split)
        return Dataset(data, targets)

    def __init__(self,  examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')


DataLoader = base.DataLoader
