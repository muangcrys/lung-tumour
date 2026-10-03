echo "#################################################################################"
echo "#                                RUNNING EVALUATION                             #"
echo "#################################################################################"

cd /workspace/lungISBI/lung-tumour
repo_base="/workspace/lungISBI/lung-tumour"
kfold_main="$repo_base/weights_kfold"
lpft_main="$repo_base/weights_kfold_2stage"
mstl_main="$repo_base/weights_kfold_luna16_double"

echo "=============================== Fresh ViViT (1ch)==============================="
fresh_c_parent="$kfold_main/vivit_random"
latest_fresh_c_dir=$(find "$fresh_c_parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)\

echo "Latest model directory: $latest_fresh_c_dir"

#run eval
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_fresh_c_dir" \
    --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
    --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --preprocessing "vivit_random" \
    --model_type "vivit" \
    --channels 1 \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"
echo "=============================== Fresh ViViT (3ch)==============================="
fresh_ccc_parent="$kfold_main/vivit_random_3ch"
latest_fresh_ccc_dir=$(find "$fresh_ccc_parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)\

echo "Latest model directory: $latest_fresh_ccc_dir"

#run eval
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_fresh_ccc_dir" \
    --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
    --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --preprocessing "vivit_random" \
    --model_type "vivit" \
    --channels 3 \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "=================================== FT ViViT ==================================="
ft_parent="$kfold_main/vivit_pretrained"
latest_ft_dir=$(find "$ft_parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)

echo "Latest model directory: $latest_ft_dir"

#run eval
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_ft_dir" \
    --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
    --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --preprocessing "vivit_pretrained" \
    --model_type "vivit" \
    --channels 3 \
    --batch_size 16 \
    --num_workers 8 \
    --device "cuda"

echo "================================== LPFT ViViT =================================="
lpft_parent="$lpft_main/vivit_pretrained-2stage"
latest_lpft_dir=$(find "$lpft_parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)

echo "Latest model directory: $latest_lpft_dir"

# run eval
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_lpft_dir" \
    --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
    --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --preprocessing "vivit_pretrained" \
    --model_type "vivit" \
    --channels 3 \
    --batch_size 16 \
    --num_workers 8 \
    --plot_2_stage \
    --device "cuda"

echo "================================== MSTL ViViT =================================="
mstl_parent="$mstl_main/vivit_pretrained-luna16double"
latest_mstl_dir=$(find "$mstl_parent" -mindepth 1 -maxdepth 1 -type d \
    -printf '%p\n' \
    | sort \
    | tail -n1)

echo "Latest model directory: $latest_mstl_dir"

# run eval
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory "$latest_mstl_dir" \
    --annotation_dir "/workspace/lungISBI/lung-tumour/data/LUNA/processed/k_fold_annotations" \
    --test_annotation "/workspace/lungISBI/lung-tumour/data/LUNA/processed/SEED_4242/test_annotations.csv" \
    --image_dir "/workspace/lungISBI/lung-tumour/data/LUNA/raw/image" \
    --preprocessing "vivit_pretrained" \
    --model_type "vivit" \
    --channels 3 \
    --batch_size 16 \
    --num_workers 8 \
    --plot_2_stage \
    --plot_mode "luna16" \
    --device "cuda"

echo "#################################################################################"
echo "#                                  END EVALUATION                               #"
echo "#################################################################################"
