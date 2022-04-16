# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python3 dqn.py \
  --logdir ./log \
  --env-name sat-v0 \
  --train-problems-paths ./data/uf50-218-tvt/train \
  --eval-problems-paths ./data/uf50-218-tvt/validation \
  --lr 0.00002 \
  --bsize 64 \
  --buffer-size 20000 \
  --eps-init 1.0 \
  --eps-final 0.01 \
  --gamma 0.99 \
  --eps-decay-steps 30000 \
  --batch-updates 50000 \
  --history-len 1 \
  --init-exploration-steps 5000 \
  --step-freq 4 \
  --target-update-freq 10 \
  --loss mse  \
  --opt adam \
  --save-freq 500 \
  --grad_clip 1 \
  --grad_clip_norm_type 2 \
  --eval-freq 1000 \
  --eval-time-limit 3600  \
  --core-steps 4 \
  --expert-exploration-prob 0.0 \
  --priority_alpha 0.5 \
  --priority_beta 0.5 \
  --e2v-aggregator sum  \
  --n_hidden 1 \
  --hidden_size 64 \
  --decoder_v_out_size 32 \
  --decoder_e_out_size 1 \
  --decoder_g_out_size 1 \
  --encoder_v_out_size 32 \
  --encoder_e_out_size 32 \
  --encoder_g_out_size 32 \
  --core_v_out_size 64 \
  --core_e_out_size 64 \
  --core_g_out_size 32 \
  --activation relu \
  --penalty_size 0.1 \
  --train_time_max_decisions_allowed 500 \
  --test_time_max_decisions_allowed 500 \
  --no_max_cap_fill_buffer \
  --lr_scheduler_gamma 1 \
  --lr_scheduler_frequency 3000 \
  --independent_block_layers 0
