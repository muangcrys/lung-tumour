#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$2
WORKING_DIR="pretrained-vivit-$UNIQUE_ID-fold-$1"
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
python -u src/k_fold_vivit_training.py \
    --k $1 \
    --model_string "pretrained_vivit" \
    --seed 4242 \
    --fold_annotation_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --val_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --num_workers 2 \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 2 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

