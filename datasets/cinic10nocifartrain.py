# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable
import numpy as np
import os
from PIL import Image
import sys
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive

from datasets import base
from datasets.cifar10 import CIFAR10
from platforms.platform import get_platform



class CINIC10(torchvision.datasets.cifar.CIFAR10):
    """`CINIC10 <http://dx.doi.org/10.7488/ds/2448>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    IT COMBINES ALL SPLITS (train, test, valid) excluding CIFAR-10 train examples for training,
    and uses the CIFAR-10 train set for testing!
    """

    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"
    tgz_md5 = "6ee4d0c996905fe93221de577967a372"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(torchvision.datasets.cifar.CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if transform is None:
            transform = torchvision.transforms.Lambda(lambda img: np.array(img))

        # check that path does not contain "cifar10-train-"
        not_cifar10_train = lambda x: "cifar10-train-" not in str(x)
        if train:
            train_data = torchvision.datasets.ImageFolder(
                os.path.join(self.root, 'train'), transform=transform, is_valid_file=not_cifar10_train)
            valid_data = torchvision.datasets.ImageFolder(
                os.path.join(self.root, 'valid'), transform=transform, is_valid_file=not_cifar10_train)
            test_data = torchvision.datasets.ImageFolder(
                os.path.join(self.root, 'test'), transform=transform, is_valid_file=not_cifar10_train)
            data_and_labels = torch.utils.data.ConcatDataset([train_data, valid_data, test_data])
            assert len(data_and_labels) == Dataset.num_train_examples()
        else:
            raise ValueError("No test set defined for cinic10nocifartrain!")

        dataloader = torch.utils.data.DataLoader(data_and_labels, batch_size=len(data_and_labels), shuffle=False, num_workers=2)
        self.data, self.targets = next(iter(dataloader))
        self.data = self.data.numpy()  # note: PIL already gives order of dims as HWC
        self.targets = self.targets.numpy()

    """Suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """
    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                download_and_extract_archive(
                    self.url, self.root, filename=self.filename, md5=self.tgz_md5)
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset, base.NdarrayDataset):
    """The CINIC-10 dataset, combining train and validation, where the test set is the subset in CIFAR-10"""

    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    MEAN = [0.47889522, 0.47227842, 0.43047404]
    STD = [0.24205776, 0.23828046, 0.25874835]

    @staticmethod
    def num_train_examples(): return 90000*3 - 50000  # cinic10 size minus cifar10 train size

    @staticmethod
    def num_test_examples(): return 50000  # cifar10 train size

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_data(train):
        if train:
            dataset = CINIC10(train=True, root=os.path.join(
                get_platform().dataset_root, 'cinic10'), download=get_platform().download_data)
        else:  # note this loads CIFAR10 train set as test!
            dataset = CIFAR10(train=True, root=os.path.join(
                get_platform().dataset_root, 'cifar10'), download=get_platform().download_data)
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
                                      [torchvision.transforms.Normalize(mean=self.MEAN, std=self.STD)])

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
