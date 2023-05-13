# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
from pruning import base
from pruning.mask import Mask

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models import base as models_base


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Layerwise Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _pruning_layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: 'models_base.Model', current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Make the number of pruned weights equal to sparse_global
        unpruned_weights = np.sum([np.sum(v) for v in current_mask.values()])

        # Find per-layer pruning fraction after removing ignored layers
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        prunable_weights = np.sum([np.sum(v) for k, v in current_mask.items() if k in prunable_tensors])
        per_layer_fraction = pruning_hparams.pruning_fraction * unpruned_weights / prunable_weights

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                for k, v in trained_model.state_dict().items()
                if k in prunable_tensors}

        # Prune the weights by layer
        new_mask = {}
        for k, v in weights.items():
            n_to_prune = np.ceil(per_layer_fraction * np.sum(current_mask[k])).astype(int)
            threshold = np.sort(np.abs(v.flatten()))[n_to_prune]
            new_mask[k] = np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
        new_mask = Mask(new_mask)
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
