# prune to equivalent sparsity with one-shot
from json import load

from lottery.branch import base
from pruning.pruned_model import PrunedModel
from training import train

from foundations import paths
from pruning.sparse_global import PruningHparams as SparseGlobalPruningHparams
from pruning.sparse_global import Strategy as SparseGlobalPruningStrategy
from utils.branch_utils import load_dense_model

class Branch(base.Branch):
    def branch_function(self,
                        start_at: str = 'rewind',
                        layers_to_ignore: str = ''):

        # get equivalent sparsity
        with open(paths.sparsity_report(self.level_root), 'r') as f:
            sparsity_info = load(f)
        pruning_fraction = 1. - sparsity_info["unpruned"] / sparsity_info["total"]

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

        dense_model = load_dense_model(self, state_step)
        # get one-shot magnitude pruning mask
        pruning_hparams = SparseGlobalPruningHparams(pruning_strategy="sparse_global",
                                                     pruning_fraction=pruning_fraction,
                                                     pruning_layers_to_ignore=layers_to_ignore)
        mask = SparseGlobalPruningStrategy.prune(pruning_hparams, dense_model)
        # Save the new mask.
        mask.save(self.branch_root)

        # Train the model with the new mask.
        model = PrunedModel(dense_model, mask)
        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "One-shot magnitude pruning to the same sparsity level as IMP iterations."

    @staticmethod
    def name():
        return 'one_shot'
