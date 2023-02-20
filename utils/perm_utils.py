import json
import torch
import numpy as np

from foundations.paths import perm


def permute_state_dict(state_dict, perm_file):
    permutation_info = torch.load(perm_file)
    perm_dict = permutation_info["permutations"]
    perm_to_axes = permutation_info["perm_to_axes"]
    for p, permutation in perm_dict.items():
        for layer_name, dim in perm_to_axes[p]:
            # not all layers have permutations (None), some layers will not be present in mask (e.g. biases)
            if permutation is not None and layer_name in state_dict:
                assert state_dict[layer_name].shape[dim] == len(permutation), (state_dict[layer_name].shape, len(permutation))
                state_dict[layer_name] = torch.tensor(np.take(state_dict[layer_name].numpy(), permutation, axis=dim))
    return state_dict


def save_perm_info(root, path_to_mask, path_to_permutation):
    # Save a file listing paths to the original mask and permutation files
    with open(perm(root), 'w') as fp:
        fp.write(json.dumps({
                "pruning_mask": str(path_to_mask),
                "pruning_permutation": str(path_to_permutation),
            }, indent=4))
