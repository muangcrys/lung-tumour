#!/bin/bash
cd /workspace/lungISBI/lung-tumour

# timestamp
UNIQUE_ID_FRESH=$(date +%Y%m%d%H%M%S)

for i in {2..4}
do
    python -u src/k_fold_vivit_training.py \
        --k "$i" \
        --model_string "vivit_random" \
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
        --time_stamp "$UNIQUE_ID_FRESH"
done

echo "Finished training fresh ViViT"

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

echo "Finished training LPFT ViViT"

# timestamp
UNIQUE_ID_MSTL=$(date +%Y%m%d%H%M%S)

for i in {1..4}
do
    python -u src/k_fold_luna16_training_vivit.py \
        --model_string "vivit_pretrained" \
        --k "$i" \
        --seed 4242 \
        --luna16_fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA16/processed/k_fold_annotations" \
        --luna16_train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA16/images" \
        --luna16_validate_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA16/images" \
        --luna25_fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --luna25_train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --luna25_validate_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --first_stage_epochs 60 \
        --second_stage_epochs 30 \
        --lr1 5e-5 \
        --lr2 5e-5 \
        --decay1 1e-3 \
        --decay2 1e-3 \
        --report_frequency 5 \
        --epochs 50 \
        --batch_size 8 \
        --metric auroc \
        --num_workers 8 \
        --device "cuda" \
        --time_stamp "$UNIQUE_ID_MSTL" 
done

echo "Finished training LPFT ViViT"
