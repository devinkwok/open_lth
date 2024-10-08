# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lottery.branch.base import Branch
from lottery.branch import randomly_prune
from lottery.branch import randomly_reinitialize
from lottery.branch import retrain
from lottery.branch import transport_mask
from lottery.branch import one_shot
from lottery.branch import retrain_one_shot
from lottery.branch import perturb_retrain

registered_branches = {
    'randomly_prune': randomly_prune.Branch,
    'randomly_reinitialize': randomly_reinitialize.Branch,
    'retrain': retrain.Branch,
    'transport_mask': transport_mask.Branch,
    'one_shot': one_shot.Branch,
    'retrain_one_shot': retrain_one_shot.Branch,
    'perturb_retrain': perturb_retrain.Branch,
}


def get(branch_name: str) -> Branch:
    if branch_name not in registered_branches:
        raise ValueError('No such branch: {}'.format(branch_name))
    else:
        return registered_branches[branch_name]
