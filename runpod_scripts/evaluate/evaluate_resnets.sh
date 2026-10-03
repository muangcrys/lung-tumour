#!/bin/bash

echo "#################################################################################"
echo "#                                RUNNING EVALUATION                             #"
echo "#################################################################################"

cd /workspace/lungISBI/lung-tumour
repo_base="/workspace/lungISBI/lung-tumour"
kfold_main="$repo_base/weights_kfold"
lpft_main="$repo_base/weights_kfold_2stage"
mstl_main="$repo_base/weights_kfold_luna16_double"

echo "=============================== MEDICALNET ==============================="

for depth in 18 34 50; do
    # fresh
    echo "Finding latest model directory for fresh medicalnet-$depth ..."
    fresh_parent="$kfold_main/fresh-medicalnet-$depth"
    latest_fresh_dir=$(find "$fresh_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_fresh_dir"

    # run evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
        --fold_directory "$latest_fresh_dir" \
        --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
        --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --preprocessing "random_init" \
        --model_type "medicalnet" \
        --depth "$depth" \
        --channels 1 \
        --batch_size 16 \
        --num_workers 8 \
        --device "cuda"

    echo "DONE evaluating fresh medicalnet-$depth"

    # pretrained
    echo "Finding latest model directory for FT medicalnet-$depth ..."
    ft_parent="$kfold_main/medicalnet-$depth"
    latest_ft_dir=$(find "$ft_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_ft_dir"

    # run evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
        --fold_directory "$latest_ft_dir" \
        --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
        --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --preprocessing "medical_pretrained" \
        --model_type "medicalnet" \
        --depth "$depth" \
        --channels 1 \
        --batch_size 16 \
        --num_workers 8 \
        --device "cuda"

    echo "DONE evaluating FT medicalnet-$depth"

    # LPFT
    echo "Finding latest model directory for LPFT medicalnet-$depth ..."
    lpft_parent="$lpft_main/medicalnet-$depth-2stage"
    latest_lpft_dir=$(find "$lpft_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_lpft_dir"

    # run evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
        --fold_directory "$latest_lpft_dir" \
        --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
        --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --preprocessing "medical_pretrained" \
        --model_type "medicalnet" \
        --depth "$depth" \
        --channels 1 \
        --batch_size 16 \
        --num_workers 8 \
        --plot_2_stage \
        --device "cuda"

    echo "DONE evaluating LPFT medicalnet-$depth"

    # MSTL
    echo "Finding latest model directory for MSTL medicalnet-$depth ..."
    mstl_parent="$mstl_main/medicalnet-$depth-luna16double"
    latest_mstl_dir=$(find "$mstl_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_mstl_dir"

    # evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_mstl_dir" \
    --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
    --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --preprocessing "medical_pretrained" \
    --model_type "medicalnet" \
    --depth "$depth" \
    --channels 1 \
    --batch_size 16 \
    --num_workers 8 \
    --plot_2_stage \
    --plot_mode "luna16" \
    --device "cuda"

    echo "DONE evaluating MSTL medicalnet-$depth"
done


echo "=============================== RESNET ==============================="

for depth in 18 34 50; do
    # fresh
    for channels in 1 3; do
        echo "Finding latest model directory for fresh resnet3d-$depth-${channels}ch ..."
        fresh_parent="$kfold_main/fresh-resnet3d-$depth-${channels}ch"
        latest_fresh_dir=$(find "$fresh_parent" -mindepth 1 -maxdepth 1 -type d \
            -printf '%p\n' \
            | sort \
            | tail -n1)
        echo "Latest model directory: $latest_fresh_dir"

        # run evaluation script
        python -u src/k_fold_evaluate_model_dir.py \
            --fold_directory "$latest_fresh_dir" \
            --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
            --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
            --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
            --preprocessing "random_init" \
            --model_type "resnet3d" \
            --depth "$depth" \
            --channels "$channels" \
            --batch_size 16 \
            --num_workers 8 \
            --device "cuda"

        echo "DONE evaluating fresh resnet3d-$depth-${channels}ch"
    done

    # pretrained
    echo "Finding latest model directory for FT resnet3d-$depth ..."
    ft_parent="$kfold_main/resnet3d-$depth"
    latest_ft_dir=$(find "$ft_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_ft_dir"

    # run evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
        --fold_directory "$latest_ft_dir" \
        --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
        --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --preprocessing "video_pretrained" \
        --model_type "resnet3d" \
        --depth "$depth" \
        --channels 3 \
        --batch_size 16 \
        --num_workers 8 \
        --device "cuda"
    
    echo "DONE evaluating FT resnet3d-$depth"

    # LPFT
    echo "Finding latest model directory for LPFT resnet3d-$depth ..."
    lpft_parent="$lpft_main/resnet3d-$depth-2stage"
    latest_lpft_dir=$(find "$lpft_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_lpft_dir"

    # run evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
        --fold_directory "$latest_lpft_dir" \
        --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
        --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --preprocessing "video_pretrained" \
        --model_type "resnet3d" \
        --depth "$depth" \
        --channels 3 \
        --batch_size 16 \
        --num_workers 8 \
        --plot_2_stage \
        --device "cuda"

    echo "DONE evaluating LPFT resnet3d-$depth"


    # MSTL
    echo "Finding latest model directory for MSTL resnet3d-$depth ..."
    mstl_parent="$mstl_main/resnet3d-$depth-luna16double"
    latest_mstl_dir=$(find "$mstl_parent" -mindepth 1 -maxdepth 1 -type d \
        -printf '%p\n' \
        | sort \
        | tail -n1)
    echo "Latest model directory: $latest_mstl_dir"

    # evaluation script
    python -u src/k_fold_evaluate_model_dir.py \
        --fold_directory "$latest_mstl_dir" \
        --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
        --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
        --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
        --preprocessing "video_pretrained" \
        --model_type "resnet3d" \
        --depth "$depth" \
        --channels 3 \
        --batch_size 16 \
        --num_workers 8 \
        --plot_2_stage \
        --plot_mode "luna16" \
        --device "cuda"

    echo "DONE evaluating MSTL resnet3d-$depth"
done

echo "#################################################################################"
echo "#                                  END EVALUATION                               #"
echo "#################################################################################"
