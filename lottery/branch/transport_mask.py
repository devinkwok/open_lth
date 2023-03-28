# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path

from lottery.branch import base
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.perm_utils import permute_state_dict, save_perm_info
from utils.file_utils import get_file_in_another_level
from utils.branch_utils import load_dense_model, get_output_layers


class Branch(base.Branch):
    def branch_function(self, pruning_mask_source_file: str,
                        pruning_permutation: str = None,
                        pruning_infer_mask_from: str = "level",
                        pruning_infer_permutation_from: str = "level",
                        start_at: str = 'rewind',
                        layers_to_ignore: str = '',
                        reinit_outputs: bool = False,
    ):
        # load mask of same level from source_model_dir
        mask_file = get_file_in_another_level(Path(pruning_mask_source_file), self.level, pruning_infer_mask_from)
        mask = Mask.load(mask_file.parent)  # strip out mask.pth
        # permute mask
        perm_file = pruning_permutation
        if perm_file is not None:
            perm_file = get_file_in_another_level(Path(pruning_permutation), self.level, pruning_infer_permutation_from)
            mask = permute_state_dict(mask, perm_file)
        # Reset the masks of any layers that shouldn't be pruned.
        if layers_to_ignore:
            for k in layers_to_ignore.split(','): mask[k] = torch.ones_like(mask[k])
        # Save the new mask.
        mask.save(self.branch_root)
        # Save a file listing paths to the original mask and permutation files
        save_perm_info(self.branch_root, mask_file, perm_file)

        # Determine the start step.
        if start_at == 'init':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = start_step
        elif start_at == 'end':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = self.lottery_desc.train_end_step
        elif start_at == 'rewind':
            start_step = self.lottery_desc.train_start_step
            state_step = start_step
        else:
            raise ValueError(f'Invalid starting point {start_at}')

        # Train the model with the new mask.
        dense_model = load_dense_model(self, state_step)

        # reinitialize mask at output layers
        if reinit_outputs:
            output_layers = get_output_layers(self)
            for k in output_layers:
                if k in mask:
                    mask[k] = torch.ones_like(dense_model.state_dict()[k])
        else:  # if the output sizes differ, must reinitialize output layers
            if any([mask[k].shape != dense_model.state_dict()[k].shape for k in mask.keys()]):
                raise ValueError(f'Sizes differ between mask and model, set --reinit_outputs?')

        model = PrunedModel(dense_model, mask)
        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Prune the model with masks from another run, applying an optional permutation."

    @staticmethod
    def name():
        return 'transport_mask'
