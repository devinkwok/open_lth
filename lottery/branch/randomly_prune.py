# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lottery.branch import base
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.branch_utils import load_dense_model


class Branch(base.Branch):
    def branch_function(self, seed: int, strategy: str = 'layerwise', start_at: str = 'rewind',
                        layers_to_ignore: str = ''):
        # Randomize the mask.
        mask = Mask.load(self.level_root)
        mask = mask.randomize(seed, strategy)
        mask.reset(layers_to_ignore)

        # Save the new mask.
        mask.save(self.branch_root)

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
        model = PrunedModel(dense_model, mask)
        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly prune the model."

    @staticmethod
    def name():
        return 'randomly_prune'
