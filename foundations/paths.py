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


def hparams_from_exp_root(root): 
    matching_files = root.rglob("hparams.log")
    try:  # find first available file in save_dir matching file_name
        file = next(matching_files)
        return file
    except:
        raise RuntimeError(f"Hparam file not found in {root}.")


def file_to_exp_root(file):
    # find experiment parent directory of file
    for path in Path(file).parents:
        if path.stem.startswith("train_") or path.stem.startswith("lottery_"):
            return path
    raise RuntimeError(f"Experiment directory containing {file} not found.")


def auto_hparams(file_or_root):
    path = Path(file_or_root) / "PLACEHOLDER_DOES_NOT_MATCH_ANY_FILE_PATH"
    exp_root = file_to_exp_root(path)
    return hparams_from_exp_root(exp_root)
