#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=test-metrics-%j.out
#SBATCH --error=test-metrics-%j.err

source ./slurm-setup.sh cifar10

STEPS="0ep,0ep200it-1ep100it@100it,2ep"

python open_lth.py lottery  \
    --default_hparams=cifar_resnet_14_8  \
    --dataset_name=cifar10  \
    --replicate=1  \
    --levels=0  \
    --training_steps=2ep  \
    --data_order_seed=42  \
    --save_ckpt_steps=$STEPS  \
    --metrics_n_train=50000  \
    --pointwise_metrics_steps=$STEPS  \
    --grad_metrics_steps=$STEPS  \
    --grad_metrics_batch_size=40  \
    --batch_forget_track  \
