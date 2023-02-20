# run using python -m make_run_hparams_table
from pathlib import Path
import pandas as pd

from platforms.platform import get_platform
from foundations.paths import auto_hparams, hparam_table
from foundations.hparams import load_hparams_from_file
from utils.file_utils import get_training_log


def canonical_dict(ckpt_dir):
    nested_hparams = load_hparams_from_file(auto_hparams(ckpt_dir))
    flat_hparams = {}
    for category, hparam_dict in nested_hparams.items():
        for k, v in hparam_dict.items():
            flat_hparams[f"{category}.{k}"] = v
    return flat_hparams

def log_info(ckpt_dir):
    log_info = {"Logger.last_" + k: v[-1] for k, v in get_training_log(ckpt_dir).items()}
    return log_info


ckpt_root = Path(get_platform().root)
save_file = hparam_table(ckpt_root)

print(f"Copying hparams for all runs in {ckpt_root}")
hparams = []
for subdir in ckpt_root.glob("*/"):
    if subdir.is_dir():
        print(subdir)
        hparams.append({'Path': subdir, **canonical_dict(subdir), **log_info(subdir)})

df = pd.DataFrame(hparams)
df.to_csv(save_file)
print(f"Hparams saved to {save_file}")
