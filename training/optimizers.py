# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import numpy as np
import torch

from foundations.hparams import TrainingHparams
from foundations.step import Step
from models.base import Model


def get_optimizer(training_hparams: TrainingHparams, model: Model) -> torch.optim.Optimizer:
    if training_hparams.optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=training_hparams.lr,
            momentum=training_hparams.momentum or training_hparams.nesterov_momentum or 0,
            weight_decay=training_hparams.weight_decay or 0,
            nesterov=training_hparams.nesterov_momentum is not None and training_hparams.nesterov_momentum > 0
        )
    elif training_hparams.optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=training_hparams.lr,
            weight_decay=training_hparams.weight_decay or 0
        )

    raise ValueError('No such optimizer: {}'.format(training_hparams.optimizer_name))


def get_lr_schedule(training_hparams: TrainingHparams, optimizer: torch.optim.Optimizer, iterations_per_epoch: int, warmup_from: Step=None):

    schedules = {
        "onecycle": onecycle_lr_schedule,
        "step": step_lr_schedule,
    }
    if training_hparams.lr_schedule not in schedules:
        raise ValueError(f"Invalid lr schedule {training_hparams.lr_schedule}: options are {list(schedules.keys())}")

    return schedules[training_hparams.lr_schedule](training_hparams, optimizer, iterations_per_epoch, warmup_from)


def step_lr_schedule(training_hparams: TrainingHparams, optimizer: torch.optim.Optimizer, iterations_per_epoch: int, warmup_from: Step=None):
    lambdas = [lambda it: 1.0]

    # Drop the learning rate according to gamma at the specified milestones.
    if bool(training_hparams.gamma) != bool(training_hparams.milestone_steps):
        raise ValueError('milestones and gamma hyperparameters must both be set or not at all.')
    if training_hparams.milestone_steps:
        milestones = [Step.from_str(x, iterations_per_epoch).iteration
                      for x in training_hparams.milestone_steps.split(',')]
        lambdas.append(lambda it: training_hparams.gamma ** bisect.bisect(milestones, it))

    # Add linear learning rate warmup if specified. Start warmup at warmup_from.
    if training_hparams.warmup_steps:
        warmup_iters = Step.from_str(training_hparams.warmup_steps, iterations_per_epoch).iteration
        delay_iters = 0 if warmup_from is None else warmup_from.iteration
        lambdas.append(lambda it: max(0., min(1.0, (it - delay_iters) / warmup_iters)))

    # Combine the lambdas.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: np.product([l(it) for l in lambdas]))


def onecycle_lr_schedule(training_hparams: TrainingHparams, optimizer: torch.optim.Optimizer, iterations_per_epoch: int, warmup_from: Step=None):
    total_steps = Step.from_str(training_hparams.training_steps, iterations_per_epoch).iteration

    # Add warmup period if specified.
    warmup_ratio = 0
    if training_hparams.warmup_steps:
        warmup_iters = Step.from_str(training_hparams.warmup_steps, iterations_per_epoch).iteration
        warmup_ratio = warmup_iters / total_steps

    if warmup_from is None or warmup_from.iteration == 0:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=training_hparams.lr,
            total_steps=total_steps,
            anneal_strategy="cos",
            pct_start=warmup_ratio,
        )
    #TODO if warmup_from is set, copy lr of warmup period (0 to warmup_iters) to start at warmup_from, and set lr to 0 beforehand
    # i.e. if warmup_from=2, turn lrs=[0, 0.5, 1, 0.8, 0.6, 0.4, 0.2, 0] into lrs=[0, 0, 0, 0.5, ]
    else:
        raise NotImplementedError(f"warmup_from={warmup_from} not implemented for onecycle lr schedule")
