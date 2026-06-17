#!/bin/bash

model="fresh-resnet3d-18"

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"
echo "###################################"
echo "#       RUNNING EVALUATION        #"
echo "###################################"

echo "Finding latest model directory for $model ..."
parent="/home/s2882278/Diss/lung-tumour/weights/$model"
latest_dir=$(find "$parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)

echo "Latest model directory: $latest_dir"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="eval-$model-$UNIQUE_ID"
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
    --model_directory "$latest_dir" \
    --annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/processed/SEED_4242/validate_annotations.csv" \
    --image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA/raw/image" \
    --preprocessing "random_init" \
    --model_type "resnet3d" \
    --depth 18 \
    --channels 3 \
    --batch_size 8 \
    --num_workers 2 \
    --device "cuda"