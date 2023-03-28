# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import datasets.registry
from foundations import hparams
from foundations.step import Step
from lottery.branch import base
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.branch_utils import load_dense_model, reinitialize_output_layers


class Branch(base.Branch):
    def branch_function(
        self,
        retrain_d: hparams.DatasetHparams,
        retrain_t: hparams.TrainingHparams,
        start_at: str = 'rewind',
        layers_to_ignore: str = '',
        reinit_outputs: bool = False,
    ):
        # Determine the start step in terms of epochs relative to the NEW task
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
        # translate epoch to new Step object for start_step
        if datasets.registry.iterations_per_epoch(retrain_d) != datasets.registry.iterations_per_epoch(self.lottery_desc.dataset_hparams):
            start_step = Step.from_epoch(start_step.ep, 0, datasets.registry.iterations_per_epoch(retrain_d))

        # Get the mask and model.
        mask = Mask.load(self.level_root)
        dense_model = load_dense_model(self, state_step)

        # reinitialize output layers of state_dict and mask
        if reinit_outputs:
            dense_model, output_layers = reinitialize_output_layers(self, dense_model, datasets.registry.num_classes(retrain_d))
            for k in output_layers:
                if k in mask:
                    mask[k] = torch.ones_like(dense_model.state_dict()[k])
        else:
            if self.lottery_desc.train_outputs != datasets.registry.num_classes(retrain_d):
                raise ValueError(f'Dataset {retrain_d.dataset_name} has output size != {self.lottery_desc.train_outputs}, must set --reinit_outputs.')

        # Reset the masks of any layers that shouldn't be pruned.
        if layers_to_ignore:
            for k in layers_to_ignore.split(','): mask[k] = torch.ones_like(dense_model.state_dict()[k])

        # Save the new mask.
        mask.save(self.branch_root)

        # Train the model with the new mask.
        model = PrunedModel(dense_model, mask)
        train.standard_train(model, self.branch_root, retrain_d, retrain_t, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Retrain the model with different hyperparameters."

    @staticmethod
    def name():
        return 'retrain'
