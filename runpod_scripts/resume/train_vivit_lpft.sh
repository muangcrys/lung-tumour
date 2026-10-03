#!/bin/bash
cd /workspace/lungISBI/lung-tumour

# timestamp
UNIQUE_ID_LPFT=$(date +%Y%m%d%H%M%S)

for i in {1..4}
do
    python -u src/k_fold_2stage_vivit_training.py \
        --model_string "pretrained_vivit" \
        --k "$i" \
        --num_channels 3 \
        --seed 4242 \
        --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --time_stamp "$UNIQUE_ID_LPFT" \
        --report_frequency 5 \
        --first_stage_epochs 20 \
        --second_stage_epochs 50 \
        --lr1 1e-4 \
        --lr2 5e-5 \
        --decay1 1e-3 \
        --decay2 1e-3 \
        --metric auroc \
        --batch_size 8 \
        --num_workers 8 \
        --device "cuda"
done