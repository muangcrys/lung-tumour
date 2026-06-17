#!/bin/bash

#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="vivit-fresh-$UNIQUE_ID"
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
python -u src/train_fresh_vivit.py \
    --train_annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/SEED_4242/train_annotations.csv" \
    --train_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --val_annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/SEED_4242/validate_annotations.csv" \
    --val_image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --num_workers 2 \
    --report_frequency 5 \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

