# run using python -m make_run_hparams_table
from json import load
from pathlib import Path
import pandas as pd
from collections import defaultdict

from foundations.paths import logger, hparam_table, branch_table
from platforms.platform import get_platform
from api import get_hparams_dict


def get_training_log(save_dir: Path) -> Path:

    def assert_list_lengths_equal(dict_of_lists):
        lists = list(dict_of_lists.values())
        if not lists:  # no lists
            return True
        iterator = iter(lists)
        the_len = len(next(iterator))
        assert all(len(l) == the_len for l in iterator), (the_len, dict_of_lists)

    try:
        with open(logger(save_dir)) as f:
            log_lines = f.readlines()
    except:  # nothing logged
        return {}
    # collate log info by iteration number
    iters = []
    log_data = defaultdict(list)
    last_it = -1
    for line in log_lines:
        key, it, value = line.strip().split(",")
        if it != last_it:  # check that every type of logged value was logged for each iter
            assert_list_lengths_equal(log_data)
            iters.append(it)
            last_it = it
        log_data[key].append(value)
    assert "test_iter" not in log_data
    log_data["test_iter"] = iters
    assert_list_lengths_equal(log_data)
    return log_data


"""
Table summarizing experiment hyperparameters
"""
def canonical_dict(ckpt_dir):
    nested_hparams = get_hparams_dict(ckpt_dir)
    flat_hparams = {}
    for category, hparam_dict in nested_hparams.items():
        for k, v in hparam_dict.items():
            flat_hparams[f"{category}.{k}"] = v
    return flat_hparams


def make_hparam_table(ckpt_root):
    print(f"Summarizing hparams for all experiments in {ckpt_root}")
    hparams = []
    for subdir in ckpt_root.glob("*/"):
        if subdir.is_dir():
            print(subdir)
            # save last items in log so that it is easy to check if an experiment was run
            log_info = {"Logger.last_" + k: v[-1] for k, v in get_training_log(subdir).items()}
            hparams.append({'Path': subdir, **canonical_dict(subdir), **log_info})

    save_file = hparam_table(ckpt_root)
    df = pd.DataFrame(hparams)
    df.to_csv(save_file)
    print(f"Hparam table saved to {save_file}")


"""
Table summarizing experiment branches
"""
def get_json_info(branch_dir):
    json_info = {}
    for file in branch_dir.glob("*.json"):
        with open(file, 'r') as f:
            json_contents = load(f)
        for k, v in json_contents.items():
            assert k not in json_info, k
            json_info[k] = v
    return json_info


def make_branch_table(ckpt_root):
    info = []
    print(f"Summarizing branches for all experiments in {ckpt_root}")
    for experiment in ckpt_root.glob("*/"):
        if experiment.is_dir():
            print(experiment.stem)
            for replicate in experiment.glob("replicate_*"):
                for level in replicate.glob("level_*"):
                    for branch in level.glob("*"):
                        branch_info = {"experiment": str(experiment), "replicate": replicate.stem, "level": level.stem, "branch": branch.stem}
                        log_info = {k: v[-1] for k, v in get_training_log(branch).items()}
                        json_info = get_json_info(branch)
                        info.append({**branch_info, **log_info, **json_info})

    save_file = branch_table(ckpt_root)
    df = pd.DataFrame(info)
    df.to_csv(save_file)
    print(f"Branch table saved to {save_file}")


def make_tables():
    ckpt_root = Path(get_platform().root)
    make_hparam_table(ckpt_root)
    make_branch_table(ckpt_root)


if __name__ == "__main__":
    make_tables()
