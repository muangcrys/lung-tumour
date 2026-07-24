#!/bin/bash

depth=50
mtype="medicalnet"
model="pretrained-$mtype-$depth"

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="kf-$model-luna16-$UNIQUE_ID"
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
python -u src/k_fold_luna16_training.py \
    --model_string "${mtype}_pretrained" \
    --depth "$depth" \
    --folds 4 \
    --seed 4242 \
    --luna16_fold_annotation_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA16/processed/k_fold_annotations" \
    --luna16_train_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA16/images" \
    --luna16_validate_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA16/images" \
    --luna25_fold_annotation_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/k_fold_annotations" \
    --luna25_train_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --luna25_validate_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --first_stage_epochs 60 \
    --second_stage_epochs 30 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --num_workers 2 \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 2 \
    --metric auroc \
    --device "cuda"

