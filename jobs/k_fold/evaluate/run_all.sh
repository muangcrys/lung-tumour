#!/bin/bash

for file in $(dirname "$0")/*.sh; do
    if [ "$(basename "$file")" != "run_all.sh" ]; then
        sbatch -p Teaching --exclude=opencast,damnii[07-12],saxa --gres=gpu:1 --cpus-per-task=2 --mem=32G -t 2-00:00:00 "$file"
    fi
done
