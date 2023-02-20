# hacky ways to navigate open_lth_data directory across different experiments
from pathlib import Path
from collections import defaultdict

from foundations.paths import logger


def get_root_replicate_level_branch(file: Path):
    # assume we have structure experiment/replicate/level/branch/file
    branch = file.parent
    level = branch.parent
    assert level.stem.startswith("level_"), file
    replicate = level.parent
    assert replicate.stem.startswith("replicate_"), file
    experiment = replicate.parent
    assert experiment.stem.startswith("train_") or experiment.stem.startswith("lottery_"), file
    return experiment, replicate, level, branch


def get_file_in_another_level(file: Path, level=None, mode="level"):
    if mode == "file" or level is None:
        return file
    if mode == "level":
        # example: let file=Path('./level_pretrain/main/somefile-level_pretrain.pth'), level='2'
        # returns Path('./level_pretrain/main/somefile-level_2.pth')
        _, parent, old_level, branch = get_root_replicate_level_branch(file)
        new_level = f"level_{str(level)}"
        new_filename = file.name.replace(old_level.stem, new_level)
        return parent / new_level / branch.stem / new_filename
    else:
        raise ValueError(f'Invalid mode {mode} for choosing level, should be "file" or "level"')


def get_training_log(save_dir: Path) -> Path:

    def assert_list_lengths_equal(dict_of_lists):
        lists = list(dict_of_lists.values())
        if not lists:  # no lists
            return True
        iterator = iter(lists)
        the_len = len(next(iterator))
        assert all(len(l) == the_len for l in iterator), (the_len, dict_of_lists)

    try:
        log_file = logger(save_dir)
    except:  # nothing logged
        return {}
    with open(log_file) as f:
        log_lines = f.readlines()
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
