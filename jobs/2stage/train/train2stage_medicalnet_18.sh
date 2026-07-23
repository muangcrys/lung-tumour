#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="medicalnet18-2stage-$UNIQUE_ID"
echo "Working directory for this run: $WORKING_DIR"
bash scripts/copy_files_to_scratch.sh "$WORKING_DIR" > /dev/null

# activate conda
echo "Activating conda environment..."
source /opt/conda/bin/activate
conda activate lung-tumour

# check allocated gpu
nvidia-smi


# train model
echo "Running training script..."
python -u src/train2stage_medicalnet.py \
    --depth 18 \
    --train_annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/SEED_4242/train_annotations.csv" \
    --train_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --val_annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/SEED_4242/validate_annotations.csv" \
    --val_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --num_workers 2 \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-4 \
    --decay2 1e-4 \
    --bce_weight1 2.0 \
    --bce_weight2 2.0 \
    --metric auroc \
    --batch_size 2 \
    --device "cuda"

