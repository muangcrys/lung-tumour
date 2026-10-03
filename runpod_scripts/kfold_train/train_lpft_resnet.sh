#!/bin/bash
cd /workspace/lungISBI/lung-tumour

# python -u src/k_fold_2stage_training.py \
#     --model_string "medicalnet_2stage" \
#     --depth 18 \
#     --folds 4 \
#     --seed 4242 \
#     --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
#     --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
#     --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
#     --report_frequency 5 \
#     --first_stage_epochs 20 \
#     --second_stage_epochs 50 \
#     --lr1 1e-4 \
#     --lr2 5e-5 \
#     --decay1 1e-3 \
#     --decay2 1e-3 \
#     --metric auroc \
#     --batch_size 16 \
#     --num_workers 8 \
#     --device "cuda"

echo "Finished training LPFT Med-18"

python -u src/k_fold_2stage_training.py \
    --model_string "medicalnet_2stage" \
    --depth 34 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --metric auroc \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "Finished training LPFT Med-34"

python -u src/k_fold_2stage_training.py \
    --model_string "medicalnet_2stage" \
    --depth 50 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --metric auroc \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "Finished training LPFT Med-50"

python -u src/k_fold_2stage_training.py \
    --model_string "resnet3d_2stage" \
    --depth 18 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --metric auroc \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "Finished training LPFT ResNet3D-18"

python -u src/k_fold_2stage_training.py \
    --model_string "resnet3d_2stage" \
    --depth 34 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --metric auroc \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "Finished training LPFT ResNet3D-34"

python -u src/k_fold_2stage_training.py \
    --model_string "resnet3d_2stage" \
    --depth 50 \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --train_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --val_image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --report_frequency 5 \
    --first_stage_epochs 20 \
    --second_stage_epochs 50 \
    --lr1 1e-4 \
    --lr2 5e-5 \
    --decay1 1e-3 \
    --decay2 1e-3 \
    --metric auroc \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "Finished training LPFT ResNet3D-50"