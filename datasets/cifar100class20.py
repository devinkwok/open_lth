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


class CIFAR100(torchvision.datasets.CIFAR100):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(CIFAR100, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset, base.NdarrayDataset):
    """CIFAR-100 dataset with 20 superclass labels."""
    LABELS = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']
    COARSE_LABEL_MAP = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                            3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                            6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                            0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                            5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                            16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                            10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                            2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                            16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                            18, 1, 2, 15, 6, 0, 17, 8, 14, 13]

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 20

    @staticmethod
    def get_data(train):
        dataset = CIFAR100(train=train, root=os.path.join(
            get_platform().dataset_root, 'cifar100'), download=get_platform().download_data)
        dataset.targets = [Dataset.COARSE_LABEL_MAP[i] for i in dataset.targets]
        return dataset.data, np.array(dataset.targets)

    @staticmethod
    def get_train_set(use_augmentation, train_split=None):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        data, targets = Dataset.get_data_split(True, train_split)
        return Dataset(data, targets, augment if use_augmentation else [])

    @staticmethod
    def get_test_set(test_split=None):
        data, targets = Dataset.get_data_split(False, test_split)
        return Dataset(data, targets)

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize(
                                        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
