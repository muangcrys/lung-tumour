#!/bin/bash

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="eval-medicalnet-18-$UNIQUE_ID"
echo "Working directory for this run: $WORKING_DIR"
bash scripts/copy_files_to_scratch.sh "$WORKING_DIR" > /dev/null

# activate conda
echo "Activating conda environment..."
source /opt/conda/bin/activate
conda activate lung-tumour

# check allocated gpu
nvidia-smi


# run evaluation script
python -u src/evaluate_model_dir.py \
    --model_directory /home/s2882278/Diss/lung-tumour/weights/medicalnet-18/20260606-023732 \
    --annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/SEED_4242/validate_annotations.csv" \
    --image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --preprocessing "medical_pretrained" \
    --model_type "medicalnet" \
    --depth 18 \
    --channels 1 \
    --batch_size 8 \
    --num_workers 2 \
    --device "cuda"