#!/bin/bash
cd /workspace/lungISBI/lung-tumour

# timestamp
UNIQUE_ID_PRETRAINED=$(date +%Y%m%d%H%M%S)

for i in {1..4}
do
    python -u src/k_fold_vivit_training.py \
        --k "$i" \
        --model_string "vivit_pretrained" \
        --num_channels 3 \
        --seed 4242 \
        --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --report_frequency 5 \
        --epochs 50 \
        --batch_size 8 \
        --num_workers 8 \
        --metric auroc \
        --learning_rate 5e-5 \
        --weight_decay 1e-3 \
        --device "cuda" \
        --time_stamp "$UNIQUE_ID_PRETRAINED"
done

echo "Finished training pretrained ViViT"