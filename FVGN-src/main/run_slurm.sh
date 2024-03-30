#!/bin/bash

#SBATCH --job-name=my_train_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20-00:00:00
#SBATCH --output=/public/home/litianyu/git-repo-local/Logger/train_output_%j.txt

export PYTHONUNBUFFERED=1

BASE_TRAIN_DIR="/lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/FVM/my_FVNN"

batch_size=24
lr=0.001
load_date=""
load_index=""

CMD="python -u $BASE_TRAIN_DIR/train.py --batch_size=$batch_size --lr=$lr"

source activate meshgn-pt-tf

srun python $CMD