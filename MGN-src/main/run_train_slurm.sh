#!/bin/bash
#SBATCH --job-name=my_train_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20-00:00:00
#SBATCH --output=/public/home/litianyu/git-repo-local/Logger/train_output_%j.txt

export PYTHONUNBUFFERED=1

source activate tf1.15

source_file_path="/home/doomduke/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/run_model.py"

batch_size=2
num_training_steps=10000000
num_rollouts=100
save_tec=false
plot_boundary=true

CHK_DIR="/home/doomduke/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/chk/new_hybrid_dataset_chk/chk-full600steps-bs=2"
DATA_DIR="/mnt/h/New-Hybrid-dataset/converted_dataset_bak/origin"
ROLLOUT_PATH="/home/doomduke/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/chk/new_hybrid_dataset_chk/chk-full600steps-bs=2/rollout/rollout.pkl"

# Create directories if they do not exist
mkdir -p "$CHK_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "${ROLLOUT_PATH%/*}"

# Train for a few steps.
srun python $source_file_path --model=cfd --mode=train --checkpoint_dir=$CHK_DIR --dataset_dir=$DATA_DIR --batch_size=$batch_size --num_training_steps=$num_training_steps

# Generate a rollout trajectory
srun python $source_file_path --model=cfd --mode=eval --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --rollout_path=${ROLLOUT_PATH} --num_rollouts=$num_rollouts  --save_tec=$save_tec --plot_boundary=$plot_boundary

echo "Train run complete."
