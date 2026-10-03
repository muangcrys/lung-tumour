#!/bin/bash
cd /workspace/lungISBI/lung-tumour

# python -u src/k_fold_training.py \
#     --model_string "fresh_medicalnet" \
#     --folds 4 \
#     --seed 4242 \
#     --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
#     --depth 18 \
#     --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
#     --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
#     --report_frequency 5 \
#     --epochs 50 \
#     --batch_size 16 \
#     --metric auroc \
#     --learning_rate 5e-5 \
#     --weight_decay 1e-3 \
#     --device "cuda"

# echo "Finished training fresh Med-18"

python -u src/k_fold_training.py \
    --model_string "fresh_medicalnet" \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --depth 34 \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

echo "Finished training fresh Med-34"

python -u src/k_fold_training.py \
    --model_string "fresh_medicalnet" \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --depth 50 \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

echo "Finished training fresh Med-50"

python -u src/k_fold_training.py \
    --model_string "fresh_resnet3d" \
    --num_channels 3 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --depth 18 \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

echo "Finished training fresh ResNet3D-18"

python -u src/k_fold_training.py \
    --model_string "fresh_resnet3d" \
    --num_channels 3 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --depth 34 \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

echo "Finished training fresh ResNet3D-34"

python -u src/k_fold_training.py \
    --model_string "fresh_resnet3d" \
    --num_channels 3 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --depth 50 \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"

echo "Finished training fresh ResNet3D-50"