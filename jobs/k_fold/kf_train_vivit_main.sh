# pass all

# cd to project root
cd "/home/s2882278/Diss/lung-tumour"

# generate unique id for this run
UNIQUE_ID=$(date +%Y%m%d%H%M%S)
WORKING_DIR="pretrained-vivit-$UNIQUE_ID"
echo "Working directory for this run: $WORKING_DIR"
bash scripts/copy_files_to_scratch.sh "$WORKING_DIR" > /dev/null

echo "Launcing k-fold to slurm"

sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train_vivit_subjob.sh 1
sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train_vivit_subjob.sh 2
sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train_vivit_subjob.sh 3
sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 ./jobs/k_fold/kf_train_vivit_subjob.sh 4