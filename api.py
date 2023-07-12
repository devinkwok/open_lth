from pathlib import Path
from itertools import chain
import json
import torch
import numpy as np

from foundations.paths import hparams
from platforms.platform import get_platform
from foundations.hparams import ModelHparams, DatasetHparams
from foundations.step import Step
from models import registry as model_registry
from datasets import registry as dataset_registry


"""
File utils
"""
def _first_match_from_root(root: Path, name_to_find: str) -> Path:
    matching_files = root.rglob(name_to_find)
    try:  # find first available file in save_dir matching file_name
        file = next(matching_files)
        return file
    except:
        raise RuntimeError(f"Pattern {name_to_find} not found in {root}.")


def _get_exp_root(path: Path) -> Path:
    # find experiment parent directory of file
    for parent in chain([path], path.parents):
        if parent.stem.startswith("train_") or parent.stem.startswith("lottery_"):
            return parent
    raise RuntimeError(f"Experiment directory containing {path} not found.")


def _find_path_in_exp(path: Path, name_to_find) -> Path:
    if path.name == name_to_find:
        return path
    exp_root = _get_exp_root(path)
    return _first_match_from_root(exp_root, name_to_find)


"""
Get objects
"""
def get_hparams_dict(path: Path, branch_name="main") -> dict:
    try:  # prefer to load from .json
        path = _find_path_in_exp(path, str(hparams(branch_name)))
        with get_platform().open(path, 'r') as fp:
            return json.load(fp)
    except RuntimeError:  # deprecated: load hparams.log instead
        path = _find_path_in_exp(path, str(Path(branch_name) / "hparams.log"))
        return _parse_hparams_dict_from_log(path)

# deprecated: for loading old hparams.log files (THIS DOES NOT HANDLE NESTED HPARAMS)
def _parse_hparams_dict_from_log(path):
    with open(path, 'r') as f:
        hparam_lines = f.readlines()
    hparams_dict = {}
    for line in hparam_lines:
        line = line.strip()
        if line.endswith(" Hyperparameters"):
            # translate keys so they match .json: make lowercase, replace space with _ and shorten "hparams"
            header = line.replace(" ", "_").replace("Hyperparameters", "hparams").lower()
            hparams_dict[header] = {}
        elif line.startswith("* "):
            k, v = line[len("* "):].split(" => ")
            hparams_dict[header][k] = v
        else:
            raise ValueError(line)
    return hparams_dict


def get_dataset_hparams(path: Path) -> dict:
    dataset_hparams = get_hparams_dict(path)["dataset_hparams"]
    return DatasetHparams.create_from_dict(dataset_hparams)


def get_model_hparams(path: Path) -> dict:
    model_hparams = get_hparams_dict(path)["model_hparams"]
    return ModelHparams.create_from_dict(model_hparams)


def get_dataset(dataset_hparams: DatasetHparams):
    return dataset_registry.registered_datasets[dataset_hparams.dataset_name].Dataset


def get_dataloader(dataset_hparams: DatasetHparams, n_examples=None, train=False, batch_size=None):
    dataset_hparams.do_not_augment = True
    dataset_hparams.subset_end = n_examples
    dataset_hparams.batch_size = n_examples if batch_size is None else batch_size
    return dataset_registry.get(dataset_hparams, train=train)


def get_model(model_hparams: ModelHparams, outputs=None) -> torch.nn.Module:
    model = model_registry.get(model_hparams, outputs=outputs)
    return model.to(device=get_platform().torch_device)


def get_state_dict(ckpt: Path):
    params = torch.load(ckpt, map_location=get_platform().torch_device)
    if "model_state_dict" in params:  # ckpt includes optimizer info
        params = params["model_state_dict"]
    return params


def get_ckpt(ckpt: Path):
    dataset_hparams = get_dataset_hparams(ckpt)
    model_hparams = get_model_hparams(ckpt)
    model = get_model(model_hparams, outputs=num_classes(dataset_hparams))
    params = get_state_dict(ckpt)
    model.load_state_dict(params)
    return (model_hparams, dataset_hparams), model, params


"""
Information
"""
def get_device():
    return get_platform().device_str


def num_train_examples(dataset_hparams: DatasetHparams):
    return dataset_registry.get_dataset(dataset_hparams).num_train_examples()


def num_test_examples(dataset_hparams: DatasetHparams):
    return dataset_registry.get_dataset(dataset_hparams).num_test_examples()


def num_classes(dataset_hparams: DatasetHparams):
    return dataset_registry.num_classes(dataset_hparams)


def iterations_per_epoch(dataset_hparams: DatasetHparams):
    return dataset_registry.iterations_per_epoch(dataset_hparams)


def get_train_test_split(dataset_hparams: DatasetHparams, train):
    split = dataset_registry.get_train_test_split(dataset_hparams)
    if split is None:
        if train:
            return np.ones(num_train_examples(dataset_hparams), dtype=bool)
        else:
            return np.zeros(num_test_examples(dataset_hparams), dtype=bool)
    else:
        if train:
            return split[:num_train_examples(dataset_hparams)]
        else:
            return split[-num_test_examples(dataset_hparams):]


"""
Checkpoints
"""
def _ckpt_name_to_ep_it(name: str):
    subparts = name.split("_")
    for part in subparts:
        if "ep" in part:
            ep = int(part.split("ep")[1])
        elif "it" in part:
            it = int(part.split("it")[1])
    return ep, it


def list_checkpoints(ckpt_dir: Path):
    if not ckpt_dir.exists():
        raise ValueError(f"Training checkpoint dir not available: {ckpt_dir}")
    it_per_ep = iterations_per_epoch(get_dataset_hparams(ckpt_dir))
    ckpts = []
    for file in ckpt_dir.glob("model_*.pth"):
        try:
            ep, it = _ckpt_name_to_ep_it(file.stem)
            step = Step.from_epoch(ep, it, it_per_ep)
            ckpts.append((step, file))
        except ValueError:  # not a valid checkpoint
            print(f"{file.name} is not a valid checkpoint")
    sorted(ckpts, key=lambda x: x[0])  # order by step
    return list(zip(*ckpts))  # return as separate lists (steps, filenames)


def get_last_checkpoint(ckpt_dir: Path):
    steps, ckpts = list_checkpoints(ckpt_dir)
    return steps[-1], ckpts[-1]


def find_ckpt_by_it(replicate_dir: Path, ep_it: str, branch="main", levels=["level_pretrain", "level_0"]):
    it_per_ep = iterations_per_epoch(get_dataset_hparams(replicate_dir))
    ep, it = _ckpt_name_to_ep_it(ep_it)
    step = Step.from_epoch(ep, it, it_per_ep)
    for level in levels:
        for i, ckpt in zip(*list_checkpoints(replicate_dir / level / branch)):
            if step == i:
                return ckpt
    raise ValueError(f"Cannot find checkpoint {ep_it} in {replicate_dir}, branch {branch}, levels {levels}")
