# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from platforms import base


class Platform(base.Platform):

    @property
    def root(self):
        return Path(os.environ.get("OPEN_LTH_ROOT"))

    @property
    def dataset_root(self):
        return Path(os.environ.get("OPEN_LTH_DATASETS"))

    @property
    def imagenet_root(self):
        return Path(os.environ.get("OPEN_LTH_DATASETS")) / 'imagenet'

    @property
    def download_data(self):
        return False
