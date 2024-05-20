# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from copy import deepcopy
import torch

import datasets.registry
import models.registry
from foundations import hparams, paths
from foundations.step import Step
from lottery.branch import base
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.branch_utils import load_dense_model, reinitialize_output_layers
from utils.file_utils import save_state_dict


def batch_noise(model, dataset_hparams, training_hparams, output_location, start_step, n_steps):
    # make a copy of the model and train it for a single step with random batch order
    copy_model = deepcopy(model)
    training_hparams = deepcopy(training_hparams)
    training_hparams.data_order_seed = None
    train_loader = datasets.registry.get(dataset_hparams, train=True)
    end_step = start_step + Step.from_str(n_steps, start_step._iterations_per_epoch)
    train.train(training_hparams, copy_model, train_loader, output_location,
                start_step=start_step, end_step=end_step)
    perturbed_state = copy_model.state_dict()
    noise = {k: v - perturbed_state[k].detach().cpu() for k, v in model.state_dict().items()}
    return noise


def init_noise(model_hparams, dataset_hparams):
    new_init = models.registry.get(model_hparams, outputs=datasets.registry.num_classes(dataset_hparams))
    return new_init.state_dict()


def noise_add(a, b, scale):
    if scale < 0:
        raise ValueError(f"Additive noise scale must be positive: {scale}")
    return a + b * scale


def noise_interpolate(a, b, scale):
    if scale < 0 or scale > 1:
        raise ValueError(f"Interpolation must be between 0 and 1: {scale}")
    return (1 - scale) * a + scale * b


def perturb_model(model, noise, combine_fn, scale, output_location):
    save_state_dict(noise, paths.perturb_noise(output_location))  # save the noise
    state_dict = model.state_dict()
    perturbed_dict = {k: combine_fn(v, noise[k], scale) for k, v in state_dict.items()}
    model.load_state_dict(perturbed_dict)


class Branch(base.Branch):
    def branch_function(
        self,
        retrain_d: hparams.DatasetHparams,
        retrain_t: hparams.TrainingHparams,
        start_at: str = 'rewind',
        layers_to_ignore: str = '',
        reinit_outputs: bool = False,
        perturb_type: str = 'batch',
        perturb_scale: float = 1.,
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
            start_step = self.lottery_desc.str_to_step(start_at)
            state_step = start_step
        # translate epoch to new Step object for start_step
        if datasets.registry.iterations_per_epoch(retrain_d) != datasets.registry.iterations_per_epoch(self.lottery_desc.dataset_hparams):
            start_step = Step.from_epoch(start_step.ep, 0, datasets.registry.iterations_per_epoch(retrain_d))

        # Get the mask and model.
        mask = Mask.load(self.level_root)
        dense_model = load_dense_model(self, state_step, self.level_root)

        # perturb the model, save the noise
        perturb_source, perturb_combine, *perturb_args = perturb_type.split("_")
        if perturb_source == 'batch':
            n_steps = "1it" if len(perturb_args) == 0 else perturb_args[0]
            noise = batch_noise(dense_model, retrain_d, retrain_t, os.path.join(self.branch_root, 'batch_noise'), start_step, n_steps=n_steps)
        elif perturb_source == 'init':
            noise = init_noise(self.lottery_desc.model_hparams, retrain_d)
        else:
            raise ValueError(f'Invalid perturbation type {perturb_type}')

        if perturb_combine == 'add':
            combine_fn = noise_add
        elif perturb_combine == 'linear':
            combine_fn = noise_interpolate
        else:
            raise ValueError(f'Invalid perturbation type {perturb_type}')

        perturb_model(dense_model, noise, combine_fn, perturb_scale, self.branch_root)

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
        return "Perturb the model and retrain with different hyperparameters."

    @staticmethod
    def name():
        return 'perturb_retrain'
