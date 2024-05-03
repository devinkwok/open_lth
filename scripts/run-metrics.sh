#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=metrics-%j.out
#SBATCH --error=metrics-%j.err

MODEL=$1
DATASET=$2
REPLICATE=$3
SEED=$RANDOM

echo $MODEL $DATASET $REPLICATE $SEED

source ./slurm-setup.sh $DATASET

python open_lth.py lottery  \
    --default_hparams=$MODEL  \
    --dataset_name=$DATASET  \
    --replicate=$REPLICATE  \
    --data_order_seed=$SEED  \
    --levels=0  \
    --training_steps=160ep  \
    --save_ckpt_steps="0ep,10ep,20ep,150ep,160ep"  \
    --metrics_n_train=50000  \
    --pointwise_metrics_steps="1ep-160ep"  \
    --pointwise_metrics_batch_size=1000  \
    --grad_metrics_steps="10ep,20ep"  \
    --grad_metrics_batch_size=40  \
    --batch_forget_track  \
