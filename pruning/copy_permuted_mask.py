# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from pathlib import Path
from json import load
import numpy as np

from foundations import hparams
from pruning import base
from pruning.mask import Mask
from utils.perm_utils import permute_state_dict
from utils.file_utils import get_file_in_another_level, get_root_replicate_level_branch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models import base as models_base


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_mask_source_file: str = None
    pruning_permutation: str = None
    pruning_infer_mask_from: str = "level"
    pruning_infer_permutation_from: str = "level"

    _name = 'Hyperparameters for permutation transport of masks from another model'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_mask_source_file = 'Path to a pruning mask of the form mask.pth at any pruning level. Other pruning levels for the given branch are detected automatically.'
    _pruning_permutation = 'File containing permutation to apply to all masks. File is of the form output by nnperm.perm. If not set, no permutation is applied.'
    _pruning_infer_mask_from = 'If "file", get mask from file at pruning_mask_source_file, if "level", replace sparsity level in pruning_permutation with the mask level.'
    _pruning_infer_permutation_from = 'If "file", get permutation from file at pruning_permutation, if "level", replace sparsity level in pruning_permutation with the mask level.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def _get_nearest_sparsity_mask(path_to_mask, target_remaining_weights):
        _, source_root, _, branch = get_root_replicate_level_branch(path_to_mask)
        levels, remaining_weights = [], []
        # get sparsity of all pruning levels
        for level in source_root.glob("*"):
            filename = level / branch.stem / "sparsity_report.json"
            if filename.exists():
                with open(filename, 'r') as f:
                    sparsity_info = load(f)
                levels.append(int(level.stem.split("_")[1]))
                remaining_weights.append(sparsity_info["unpruned"])
        if len(levels) == 0:
            raise ValueError(f"No pruning masks found in {source_root} with branch {branch}.")
        # find level with closest sparsity
        remaining_weights = np.array(remaining_weights)
        sort_idx = np.flip(np.argsort(remaining_weights))
        remaining_weights = remaining_weights[sort_idx]
        closest_idx = np.argmin(np.abs(remaining_weights - target_remaining_weights))
        levels = [levels[i] for i in sort_idx]
        return levels[closest_idx]

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: 'models_base.Model', current_mask: Mask = None):
        if pruning_hparams.pruning_mask_source_file is None:
            raise ValueError("Need to specify directory for pruning_mask_source_file.")
        # infer pruning level from sparsity of current_mask
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        # load mask with next level of sparsity from source_model_dir
        next_level = 1 + Strategy._get_nearest_sparsity_mask(Path(pruning_hparams.pruning_mask_source_file), number_of_remaining_weights)
        # get mask file
        mask_file = get_file_in_another_level(
                Path(pruning_hparams.pruning_mask_source_file), next_level,
                pruning_hparams.pruning_infer_mask_from)
        new_mask = Mask.load(mask_file.parent)  # strip out mask.pth
        # permute mask according to permutation
        if pruning_hparams.pruning_permutation is not None:
            perm_file = get_file_in_another_level(
                    Path(pruning_hparams.pruning_permutation), next_level,
                    pruning_hparams.pruning_infer_permutation_from)
            new_mask = permute_state_dict(new_mask, perm_file)
        return new_mask
