"""
cinic10: standard cinic10 dataset
cinic10: standard cinic10 dataset
cinic10nocifarsubset: cinic10 *WITHOUT cifar10 train set* for training, cifar10 *train set* for testing, with the training set randomly pruned to 50000 examples (keeping all cifar10 test set examples)
"""
from pathlib import Path
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

    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    CLASS_TO_IDX = {k: i for i, k in enumerate(LABELS)}
    SPLITS = ['train', 'valid', 'test']
    SEED = 42

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        if not train:
            raise ValueError("CINIC-10 test set should not be used by cinic10nocifarsubset!")

        super(torchvision.datasets.cifar.CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        data_root = Path(self.root)

        def get_data_from_splits(splits, is_valid_file):
            data = []
            labels = []
            for split in splits:
                dataset = torchvision.datasets.ImageFolder(data_root / split, is_valid_file=is_valid_file)
                assert dataset.class_to_idx == self.CLASS_TO_IDX
                for x, y in dataset:
                    data.append(x)
                    labels.append(y)
            data = np.stack(data, axis=0)
            labels = np.array(labels)
            return data, labels

        # keep all cifar10 test data
        def is_cifar_test(x):
            return Path(x).name.startswith("cifar10-test-")
        cifar10_test_data, cifar10_test_targets = get_data_from_splits(["test"], is_valid_file=is_cifar_test)

        # prune non-cifar10 data so that total size is same as cifar10, equally per class
        def is_not_cifar(x):
            return not Path(x).name.startswith("cifar10-")
        non_cifar10_data, non_cifar10_targets = get_data_from_splits(self.SPLITS, is_valid_file=is_not_cifar)

        g = torch.Generator()
        g.manual_seed(self.SEED)

        keep_data_mask = np.zeros(len(non_cifar10_targets), dtype=bool)
        for label in self.CLASS_TO_IDX.values():
            label_idx = np.where(non_cifar10_targets == label)[0]
            # randomly pick: there are 270000 minus 60000 from cifar10, divided by 10 classes, and we want 50000 total including cifar10 test split
            keep_idx = torch.randperm(21000, generator=g)[:4000].numpy()
            keep_data_mask[label_idx[keep_idx]] = 1
        non_cifar10_data = non_cifar10_data[keep_data_mask]
        non_cifar10_targets = non_cifar10_targets[keep_data_mask]

        # combine all data
        self.data = np.concatenate([cifar10_test_data, non_cifar10_data], axis=0)
        self.targets = np.concatenate([cifar10_test_targets, non_cifar10_targets])

        # sanity checks
        assert len(self.data) == Dataset.num_train_examples()
        for label in self.CLASS_TO_IDX.values():
            assert np.count_nonzero(self.targets == label) == 5000, (label, np.count_nonzero(self.targets == label))


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
    def num_train_examples(): return 50000  # same as cifar10 after pruning

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
        #NOTE this is an error (not fixed to be consistent with old runs of CIFAR10): RandomCrop should have fill=MEAN
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        augment = []  #debug
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
