#!/bin/bash

# script for setting up slurm job
# arguments are subdirs of datasets under $DATA_SOURCE
DATA_SOURCE=$HOME/data

# load modules
module load python/3.7
module load pytorch/1.4

# install or activate requirements
if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# copy datasets to SLURM_TMPDIR
for SUBDIR in "$@"
do
    if ! [ -d "$SLURM_TMPDIR/data/$SUBDIR/" ]; then
        mkdir -p $SLURM_TMPDIR/data/$SUBDIR
        echo "Copying $SUBDIR ..."
        cp -r $DATA_SOURCE/$SUBDIR $SLURM_TMPDIR/data/
        # extract any archives
        for FILE in "$SLURM_TMPDIR/data/$SUBDIR/*.tar.gz"
        do
            tar -xzf $FILE -C $SLURM_TMPDIR/data/$SUBDIR
        done
    fi
done

# set environment variables for open_lth/paltforms/local.py
export OPEN_LTH_ROOT="$HOME/scratch/open_lth_data/"
export OPEN_LTH_DATASETS="$SLURM_TMPDIR/data/"
