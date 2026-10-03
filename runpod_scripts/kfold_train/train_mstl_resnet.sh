#!/bin/bash
cd /workspace/lungISBI/lung-tumour

python -u src/k_fold_luna16_training.py \
    --model_string "medicalnet_pretrained" \
    --depth 18 \
    --folds 4 \
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
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --device "cuda"

# echo "Finished training Med-18 on LUNA16 and LUNA25"

python -u src/k_fold_luna16_training.py \
    --model_string "medicalnet_pretrained" \
    --depth 34 \
    --folds 4 \
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
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --device "cuda"

echo "Finished training Med-34 on LUNA16 and LUNA25"

python -u src/k_fold_luna16_training.py \
    --model_string "medicalnet_pretrained" \
    --depth 50 \
    --folds 4 \
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
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --device "cuda"

echo "Finished training Med-50 on LUNA16 and LUNA25"

python -u src/k_fold_luna16_training.py \
    --model_string "resnet3d_pretrained" \
    --depth 18 \
    --folds 4 \
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
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --device "cuda"

echo "Finished training ResNet3D-18 on LUNA16 and LUNA25"

python -u src/k_fold_luna16_training.py \
    --model_string "resnet3d_pretrained" \
    --depth 34 \
    --folds 4 \
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
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --device "cuda"

echo "Finished training ResNet3D-34 on LUNA16 and LUNA25"

python -u src/k_fold_luna16_training.py \
    --model_string "resnet3d_pretrained" \
    --depth 50 \
    --folds 4 \
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
    --batch_size 16 \
    --num_workers 8 \
    --metric auroc \
    --device "cuda"

echo "Finished training ResNet3D-50 on LUNA16 and LUNA25"