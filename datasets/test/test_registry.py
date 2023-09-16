# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from datasets import base, registry, cifar10
from foundations import hparams
from testing import test_case


class TestRegistry(test_case.TestCase):
    def setUp(self):
        super(TestRegistry, self).setUp()
        self.dataset_hparams = hparams.DatasetHparams('cifar10', 50)

    def test_get(self):
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 1000)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

        loader = registry.get(self.dataset_hparams, train=False)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 200)

    def test_do_not_augment(self):
        self.dataset_hparams.do_not_augment = True
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_get_subsample(self):
        self.dataset_hparams.transformation_seed = 0
        self.dataset_hparams.subsample_fraction = 0.1
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 100)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_get_subset(self):
        _, all_labels = next(iter(registry.get(self.dataset_hparams, train=True)))
        even_idx = np.arange(0, 50000, 2)
        self.dataset_hparams.subset_file = "./testing/TESTING/test_get_subset.npy"
        np.save(self.dataset_hparams.subset_file, even_idx)
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 500)

        _, labels = next(iter(loader))
        np.testing.assert_array_equal(labels[:25], all_labels[np.arange(0, 50, 2)])

    def test_get_random_labels(self):
        self.dataset_hparams.transformation_seed = 0
        self.dataset_hparams.random_labels_fraction = 1.0
        self.dataset_hparams.do_not_augment = True
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 1000)

        minibatch, labels = next(iter(loader))
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_get_unsupervised_labels(self):
        self.dataset_hparams.transformation_seed = 0
        self.dataset_hparams.unsupervised_labels = 'rotation'
        loader = registry.get(self.dataset_hparams, train=True)
        self.assertIsInstance(loader, base.DataLoader)
        self.assertEqual(len(loader), 1000)

        minibatch, labels = next(iter(loader))
        self.assertEqual(np.max(labels.numpy()), 3)
        self.assertEqual(minibatch.numpy().shape[0], self.dataset_hparams.batch_size)

    def test_iterations_per_epoch(self):
        self.assertEqual(registry.iterations_per_epoch(self.dataset_hparams), 1000)
        self.dataset_hparams.subsample_fraction = 0.1
        self.assertEqual(registry.iterations_per_epoch(self.dataset_hparams), 100)

    def test_custom_train_test_split(self):
        def analyze_split(fraction, randomize, fold):
            train_test_split = base.TrainTestSplit(cifar10.Dataset, fraction, randomize, 0, fold)
            mask = train_test_split.get_train_mask()
            train_idx = np.nonzero(np.logical_not(mask[:50000]))[0]
            test_idx = np.nonzero(mask[50000:])[0]
            return np.min(train_idx), np.max(train_idx), np.min(test_idx), np.max(test_idx)

        def assert_close(num_a, num_b, tolerance=1000):
            self.assertTrue(abs(num_a - num_b) < tolerance)

        train_min, train_max, test_min, test_max = analyze_split(0.25, False, 0)
        assert_close(train_min, 0)
        assert_close(train_max, 2500)
        assert_close(test_min, 7500)
        assert_close(test_max, 10000)
        train_min, train_max, test_min, test_max = analyze_split(0.25, False, 3)
        assert_close(train_min, 7500)
        assert_close(train_max, 10000)
        assert_close(test_min, 0)
        assert_close(test_max, 2500)
        train_min, train_max, test_min, test_max = analyze_split(0.5, True, 1)
        assert_close(train_min, 0)
        assert_close(train_max, 50000)
        assert_close(test_min, 0)
        assert_close(test_max, 10000)


test_case.main()
