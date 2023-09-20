# run using python -m make_run_hparams_table
import json
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


def get_json_info(branch_dir):
    json_info = {}
    for file in branch_dir.glob("*.json"):
        with open(file, 'r') as f:
            json_info[file.stem] = json.load(f)
    return json_info


def flatten_dict(nested_hparams, prefix=None):
    flat_hparams = {}
    for k, v in nested_hparams.items():
        if prefix is not None:
            k = f"{prefix}.{k}"
        if isinstance(v, dict):
            flat_hparams = {**flat_hparams, **flatten_dict(v, prefix=k)}
        else:
            flat_hparams[k] = v
    return flat_hparams


"""
Table summarizing experiment hyperparameters
"""
def make_hparam_table(ckpt_root):
    print(f"Summarizing hparams for all experiments in {ckpt_root}")
    rows = []
    for subdir in ckpt_root.glob("*/"):
        if subdir.is_dir():
            # save last items in log so that it is easy to check if an experiment was run
            log_info = {"logger.last_" + k: v[-1] for k, v in get_training_log(subdir).items()}
            try:
                hparams_info = flatten_dict(get_hparams_dict(subdir))
                rows.append({'Path': subdir, **hparams_info, **log_info})
            except RuntimeError:
                print(f"Error loading hparams: {subdir}")

    save_file = hparam_table(ckpt_root)
    df = pd.DataFrame(rows)
    df.to_csv(save_file)
    print(f"Hparam table saved to {save_file}")


"""
Table summarizing experiment branches
"""
def make_branch_table(ckpt_root):
    info = []
    print(f"Summarizing branches for all experiments in {ckpt_root}")
    for experiment in ckpt_root.glob("*/"):
        if experiment.is_dir():
            for replicate in experiment.glob("replicate_*"):
                for level in replicate.glob("level_*"):
                    for branch in level.glob("*"):
                        branch_info = {"experiment": str(experiment), "replicate": replicate.stem, "level": level.stem, "branch": branch.stem}
                        log_info = {k: v[-1] for k, v in get_training_log(branch).items()}
                        # assume hparams in json and not log file, will be picked up by get_json_info
                        json_info = flatten_dict(get_json_info(branch))
                        # only save if branch is not empty
                        if len(log_info) == 0 and len(json_info) == 0:
                            print(f"Empty branch: {branch}")
                        else:
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
