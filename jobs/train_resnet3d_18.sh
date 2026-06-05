#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
bash scripts/copy_files_to_scratch.sh

# activate conda
source /opt/conda/bin/activate
conda activate lung-tumour

# check allocated model
nvidia-smi

# run script
python -u src/train_resnet3d_18.py \
    --data_dir "/disk/scratch/s2882278/lung-tumour/data" \
    --num_epochs 10 \


