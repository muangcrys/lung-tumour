echo "#################################################################################"
echo "#                                RUNNING EVALUATION                             #"
echo "#################################################################################"

cd /workspace/lungISBI/lung-tumour
repo_base="/workspace/lungISBI/lung-tumour"
kfold_main="$repo_base/weights_kfold"
lpft_main="$repo_base/weights_kfold_2stage"
mstl_main="$repo_base/weights_kfold_luna16_double"
depth=50

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