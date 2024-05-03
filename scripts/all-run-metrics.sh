#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=gen-metrics-%j.out
#SBATCH --error=gen-metrics-%j.err

# MODEL=(cifar_resnet_20 cifar_vgg_16  \
#    cifar_resnet_20_8 cifar_resnet_20_32 cifar_vgg_16_32 cifar_vgg_16_128)  \
#    cifar_resnet_34 cifar_resnet_50 cifar_vgg_13 cifar_vgg_19  \
#)
MODEL=(cifar_resnet_20_64 cifar_vgg_16_16  \
       cifar_resnet_32 cifar_vgg_11)
DATASET=(cifar10 cifar100)
# REPLICATE=($(seq 1 1 50))
REPLICATE=($(seq 1 1 5))

parallel --delay=5 --jobs=1  \
    sbatch ./scripts/run-metrics.sh {1} {2} {3} {4}  \
    ::: ${MODEL[@]}  \
    ::: ${DATASET[@]}  \
    ::: ${REPLICATE[@]}  \
