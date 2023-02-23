# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from platforms import base


class Platform(base.Platform):

    @property
    def root(self):
        return Path(os.environ.get("HOME")) / 'scratch/open_lth_data/'

    @property
    def dataset_root(self):
        return Path(os.environ.get("SLURM_TMPDIR")) / 'data'

    @property
    def imagenet_root(self):
        return Path(os.environ.get("SLURM_TMPDIR")) / 'data/imagenet'

    @property
    def download_data(self):
        return False
