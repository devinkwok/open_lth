# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


def checkpoint(root): return Path(root) / 'checkpoint.pth'


def logger(root): return Path(root) / 'logger'


def mask(root): return Path(root) / 'mask.pth'


def sparsity_report(root): return Path(root) / 'sparsity_report.json'


def model(root, step): return Path(root) / 'model_ep{}_it{}.pth'.format(step.ep, step.it)


def hparams(root): return Path(root) / 'hparams.log'


def perm(root): return Path(root) / "perm_mask_source.json"


def branch_table(root): return Path(root) / "exp-branches.csv"


def hparam_table(root): return Path(root) / "exp-hparams.csv"
