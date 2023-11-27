#!/bin/bash
export PYTHONUNBUFFERED=1

source activate tf1.15

source_file_path="FVGN-sub-Meshgraphnet/MGN-src/main/run_model.py"

batch_size=2
num_training_steps=10000000
num_rollouts=100
dual_edge=true
save_tec=false
plot_boundary=true

CHK_DIR="Logger/Hybriddataset-full600-dualedge-bs=2/chk"
DATA_DIR=""
ROLLOUT_PATH="Logger/Hybriddataset-full600-dualedge-bs=2/rollout/rollout.pkl"

# Create directories if they do not exist
mkdir -p "$CHK_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "${ROLLOUT_PATH%/*}"

# Train for a few steps.
python $source_file_path --model=cfd --mode=train --checkpoint_dir=$CHK_DIR --dataset_dir=$DATA_DIR --batch_size=$batch_size --num_training_steps=$num_training_steps --dual_edge=$dual_edge

# Generate a rollout trajectory
python $source_file_path --model=cfd --mode=eval --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --rollout_path=${ROLLOUT_PATH} --num_rollouts=$num_rollouts  --save_tec=$save_tec --plot_boundary=$plot_boundary --dual_edge=$dual_edge

echo "Train run complete."

#sh /lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/run_train.sh 2>&1 | tee /lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/Hybriddataset-full600-dualedge-bs=2/training_log.txt
