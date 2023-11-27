#!/bin/bash
export PYTHONUNBUFFERED=1

source activate tf1.15

source_file_path="/lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/run_model.py"

batch_size=2
num_training_steps=10000000
num_rollouts=100
save_tec=false
plot_boundary=true

CHK_DIR="/lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/chk/new_hybrid_dataset_chk/chk-full600steps-bs=2"
DATA_DIR="/lvm_data/litianyu/dataset/MeshGN/new_hybrid_dataset_Re=200-1500/origin"
ROLLOUT_PATH="/lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/chk/new_hybrid_dataset_chk/chk-full600steps-bs=2/rollout/rollout.pkl"

# Create directories if they do not exist
mkdir -p "$CHK_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "${ROLLOUT_PATH%/*}"

# Generate a rollout trajectory
python $source_file_path --model=cfd --mode=eval --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --rollout_path=${ROLLOUT_PATH} --num_rollouts=$num_rollouts  --save_tec=$save_tec --plot_boundary=$plot_boundary

echo "Test run complete."
