# retrieve all metrics that have the same hparams other than data_order_seed
# save checkpoints and final metrics, but ignore the metric checkpoints
import argparse
import shutil
from pathlib import Path
import json

import api
from lottery.desc import LotteryDesc


def replace_hparams(training_hparams, replace_dict):
    replaced = {}
    for k, v in replace_dict.items():
        replaced[k] = training_hparams[k]
        training_hparams[k] = v
    return training_hparams, replaced


def collate_metrics(save_location, patterns, dry_run):
    platform = api.get_platform()

    for exp in platform.root.glob("lottery_*"):
        try:
            hparams_dict = api.get_hparams_dict(exp)
        except:
            print(f"Skipping {exp}, cannot find hparams file")
            continue
        # skip if experiment has no metrics
        if "batch_forget_track" not in hparams_dict["training_hparams"]:
            continue
        if not hparams_dict["training_hparams"]["batch_forget_track"]:
            continue

        # combine to same hparams other than data_order_seed
        replace_values = {
            "data_order_seed": None,
            "pointwise_metrics_batch_size": None,
            "grad_metrics_batch_size": None,
        }
        training_hparams, replaced_hparams = replace_hparams(hparams_dict["training_hparams"], replace_values)
        hparams_dict["training_hparams"] = training_hparams
        dataset_hparams = api.get_dataset_hparams(hparams_dict)
        model_hparams = api.get_model_hparams(hparams_dict)
        training_hparams = api.get_training_hparams(hparams_dict)
        pruning_hparams = api.get_pruning_hparams(hparams_dict)
        desc = LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
        save_dir = save_location / desc.hashname
        if not platform.exists(save_dir):
            platform.makedirs(save_dir)

        for source_dir in exp.glob("*"):
            target_dir = save_dir / source_dir.name
            # don't overwrite existing
            if target_dir.exists():
                print(f"{target_dir} already exists, skipping")
                continue
            # copy over all files matching patterns
            for pattern in patterns:
                for source_file in source_dir.rglob(pattern):
                    target_file = target_dir / source_file.relative_to(source_dir)
                    print(f"Copying {source_file} to {target_file}")
                    if not dry_run:
                        if not platform.exists(target_file.parent):
                            platform.makedirs(target_file.parent)
                        shutil.copy(source_file, target_file, follow_symlinks=False)
            # save combined hparams, and a separate file indicating original location
            hparam_location = target_dir / "level_0" / "main"
            print(f"Saving hparam files to {hparam_location}")
            if not dry_run:
                with open(hparam_location / 'hparams.json', 'w') as file:
                    json.dump(hparams_dict, file)
                with open(hparam_location / 'copied_from.json', 'w') as file:
                    json.dump({"original_location": str(exp), **replaced_hparams}, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, type=Path)
    parser.add_argument('--dry_run', default=False, action="store_true")
    args = parser.parse_args()

    # save checkpoints and final metrics, but ignore the metric checkpoints
    patterns = ["model_*.pth", "*_*.npz"]
    collate_metrics(args.save_dir, patterns, args.dry_run)
