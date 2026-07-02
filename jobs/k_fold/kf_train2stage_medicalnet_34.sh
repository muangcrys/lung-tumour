#!/bin/bash

#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="kf-medicalnet34-2stage-$UNIQUE_ID"
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
python -u src/k_fold_2stage_training.py \
    --model_string "medicalnet_2stage" \
    --depth 34 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --val_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --num_workers 2 \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --metric auroc \
    --batch_size 2 \
    --device "cuda"

