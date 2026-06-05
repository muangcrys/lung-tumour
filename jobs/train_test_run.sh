#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
bash scripts/copy_files_to_scratch.sh

# activate conda
source /opt/conda/bin/activate
conda activate lung-tumour


# train model
python -u src/train_resnet3d.py \
    --depth 18 \
    --train_annotation "/disk/scratch/s2882278/lung-tumour/data/LUNA/processed/SEED_4242/train_annotations.csv" \
    --train_image_dir "/disk/scratch/s2882278/lung-tumour/data/LUNA/raw/image" \
    --val_annotation "/disk/scratch/s2882278/lung-tumour/data/LUNA/processed/SEED_4242/validate_annotations.csv" \
    --val_image_dir "/disk/scratch/s2882278/lung-tumour/data/LUNA/raw/image" \
    --num_workers 2 \
    --report_frequency 5 \
    --num_epochs 10 \
    --batch_size 16 \
    --device "cuda"

