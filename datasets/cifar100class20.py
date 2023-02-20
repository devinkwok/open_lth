#TODO fix

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datasets import base, cifar100


class Dataset(cifar100.Dataset):
    """CIFAR-100 dataset with 20 superclass labels."""
    COARSE_LABELS = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']
    SUPERCLASS_LABEL_MAP = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
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
    def num_classes(): return 20


DataLoader = base.DataLoader
