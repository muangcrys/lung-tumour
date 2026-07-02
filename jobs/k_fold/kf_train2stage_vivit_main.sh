#!/bin/bash
# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
echo "Unique ID for this batch: $UNIQUE_ID"

echo "Launcing k-fold to slurm"

sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train2stage_vivit_subjob 1 $UNIQUE_ID
sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train2stage_vivit_subjob 2 $UNIQUE_ID
sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train2stage_vivit_subjob 3 $UNIQUE_ID
sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train2stage_vivit_subjob 4 $UNIQUE_ID

echo "DONE!"