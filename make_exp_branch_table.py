# run using python -m make_branch_test_table
from json import load
from pathlib import Path
import pandas as pd

from platforms.platform import get_platform
from utils.file_utils import get_training_log
from foundations.paths import branch_table


def get_json_info(branch_dir):
    json_info = {}
    for file in branch_dir.glob("*.json"):
        with open(file, 'r') as f:
            json_contents = load(f)
        for k, v in json_contents.items():
            assert k not in json_info, k
            json_info[k] = v
    return json_info


info = []
print("Finding all branches.")
ckpt_root = Path(get_platform().root)
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
print(f"Test acc saved to {save_file}")
