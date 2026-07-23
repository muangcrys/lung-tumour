#!/bin/bash

depth=50
mtype="resnet3d"
channels=1
model="fresh-$mtype-$depth-${channels}ch"

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"
echo "#################################################################################"
echo "#                                RUNNING EVALUATION                             #"
echo "#################################################################################"

echo "Finding latest model directory for $model ..."
parent="/home/s2882278/Diss/lung-tumour/weights_kfold/$model"
latest_dir=$(find "$parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)

echo "Latest model directory: $latest_dir"

# copy files to scratch
echo "Copying files to scratch..."
# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="evalkf-luna16-$model-$UNIQUE_ID"
echo "Working directory for this run: $WORKING_DIR"
bash scripts/copy_files_to_scratch.sh "$WORKING_DIR" > /dev/null

# activate conda
echo "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate lung-tumour

# check allocated gpu
nvidia-smi

# run evaluation script
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_dir" \
    --annotation_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA16/processed/k_fold_annotations" \
    --test_annotation "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA16/processed/annotation.csv" \
    --image_dir "/disk/scratch/s2882278/lung-tumour/$WORKING_DIR/data/LUNA16/images" \
    --save_prefix "luna16" \
    --skip_plot_loss \
    --skip_plot_metrics \
    --preprocessing "random_init" \
    --model_type "$mtype" \
    --depth "$depth" \
    --channels "$channels" \
    --batch_size 8 \
    --num_workers 2 \
    --device "cuda"

