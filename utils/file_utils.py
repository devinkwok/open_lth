from pathlib import Path
from platforms.platform import get_platform


def save_state_dict(state_dict, output_file):
    if not get_platform().is_primary_process: return
    if not get_platform().exists(output_file.parent): get_platform().makedirs(output_file.parent)
    get_platform().save_model(state_dict, output_file)


"""hacky ways to navigate open_lth_data directory across different experiments
"""
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
