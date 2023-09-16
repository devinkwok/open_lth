# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from datasets import base, cifar10, cifar10class5, cifar100, cifar100class20, cifar100class10, mnist, imagenet, svhn, eurosat, pixelpermutedcifar10
from foundations.hparams import DatasetHparams
from platforms.platform import get_platform


registered_datasets = {
    'cifar10': cifar10,
    'cifar10class5': cifar10class5,
    'cifar100': cifar100,
    'cifar100class20': cifar100class20,
    'cifar100class10': cifar100class10,
    'mnist': mnist,
    'imagenet': imagenet,
    'svhn': svhn,
    'eurosat': eurosat,
    'pixelpermutedcifar10': pixelpermutedcifar10,
}


def get_dataset(dataset_hparams: DatasetHparams):
    return registered_datasets[dataset_hparams.dataset_name].Dataset


def get_train_test_split(dataset_hparams: DatasetHparams):
    if dataset_hparams.custom_train_test_split:
        train_test_split = base.TrainTestSplit(
                get_dataset(dataset_hparams),
                dataset_hparams.train_test_split_fraction,
                dataset_hparams.split_randomize,
                (dataset_hparams.transformation_seed or 0) + 2,
                dataset_hparams.split_fold
            )  # custom train/test split
        return train_test_split.get_train_mask()
    return None


def get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters."""

    if dataset_hparams.dataset_name not in registered_datasets:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    seed = dataset_hparams.transformation_seed or 0
    data_class = get_dataset(dataset_hparams)
    # custom train/test split
    train_split = get_train_test_split(dataset_hparams)
    # Get the dataset itself.
    use_augmentation = train and not dataset_hparams.do_not_augment
    if train:
        dataset = data_class.get_train_set(use_augmentation, train_split=train_split)
    else:
        test_split = None if train_split is None else np.logical_not(train_split)
        dataset = data_class.get_test_set(test_split=test_split)

    # Transform the dataset.
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)

    # arbitrary subsample
    if train and (dataset_hparams.subset_start is not None or dataset_hparams.subset_stride != 1  \
                  or dataset_hparams.subset_end is not None or dataset_hparams.subset_file is not None):
        # allow start,end,stride when cv_fold is set
        if dataset_hparams.subsample_fraction is not None and dataset_hparams.cv_fold is None:
            raise ValueError("Cannot have both subsample_fraction and subset_[start,end,stride,file]")
        dataset.subset(_get_subset_idx(dataset_hparams))

    # random subsample
    if (train and dataset_hparams.subsample_fraction is not None) or (dataset_hparams.cv_fold is not None):
        # allow test set to be subsampled when cv_fold is set
        dataset.subsample(seed, dataset_hparams.subsample_fraction, dataset_hparams.cv_fold)

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can blur images.')
        else:
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        elif not isinstance(dataset, base.ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)

    # Create the loader.
    return registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=get_platform().num_workers)

def _get_subset_idx(dataset_hparams: DatasetHparams):
    num_train_examples = get_dataset(dataset_hparams).num_train_examples()
    if dataset_hparams.subset_file is not None:
        examples_to_retain = np.load(dataset_hparams.subset_file)
        assert len(examples_to_retain.shape) < 2 and len(examples_to_retain) <= num_train_examples and np.min(examples_to_retain) >= 0 and np.max(examples_to_retain) < num_train_examples
        return examples_to_retain
    subset_start = 0 if dataset_hparams.subset_start is None else dataset_hparams.subset_start
    subset_end = num_train_examples if dataset_hparams.subset_end is None else dataset_hparams.subset_end
    subset_stride = 1 if dataset_hparams.subset_stride is None else dataset_hparams.subset_stride
    assert subset_start >= 0 and subset_end <= num_train_examples and subset_start < subset_end  \
        and subset_stride > 0 and subset_stride <= num_train_examples
    return np.arange(subset_start, subset_end, subset_stride)

def iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = get_dataset(dataset_hparams).num_train_examples()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction
    else:
        num_train_examples = len(list(_get_subset_idx(dataset_hparams)))

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_classes = get_dataset(dataset_hparams).num_classes()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        else:
            return 4

    return num_classes
