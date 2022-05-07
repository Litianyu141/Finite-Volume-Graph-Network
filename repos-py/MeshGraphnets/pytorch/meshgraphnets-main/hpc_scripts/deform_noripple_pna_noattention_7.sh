#!/bin/bash
#SBATCH --job-name=deform_noripple_sum_noattention
#SBATCH --partition=gpu_8

i=7
srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --model=deform --mode=all --rollout_split=valid --dataset=deforming_plate  --epochs=25 --trajectories=1000 --num_rollouts=100 --core_model=encode_process_decode --message_passing_aggregator=pna --message_passing_steps=${i} --attention=False --ripple_used=False --use_prev_config=True
