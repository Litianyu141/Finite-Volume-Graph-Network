# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Functions to build evaluation metrics for CFD data."""

import tensorflow as tf
import numpy as np
from meshgraphnets.common import NodeType


def _rollout(model, initial_state, num_steps):
  """Rolls out a model trajectory."""
  node_type = initial_state['node_type'][:, 0]
  mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL),
                       tf.equal(node_type, NodeType.OUTFLOW))

  def step_fn(step, velocity, trajectory):
    prediction,predicted_pressure = model({**initial_state,
                        'velocity': velocity})
    # don't update boundary nodes
    next_velocity = tf.where(mask, prediction, velocity)
    next_uvp = tf.concat([next_velocity,predicted_pressure],axis=1)
    trajectory = trajectory.write(step, next_uvp)

    return step+1, next_velocity, trajectory

  _, _, output = tf.while_loop(
      cond=lambda step, cur, traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_state['velocity'],
                 tf.TensorArray(tf.float32, num_steps)),
      parallel_iterations=1)
  return output.stack()

  
def evaluate(model, inputs):
  """Performs model rollouts and create stats."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  num_steps = inputs['cells'].shape[0]
  prediction = _rollout(model, initial_state, num_steps)
  
  # 分离速度和压力预测
  prediction_velocity = prediction[:, :, 0:2]
  prediction_pressure = prediction[:, :, 2:3]
  
  # 创建一个时间步长的数组，用于平均化
  time_steps = tf.range(1, num_steps.value + 1, dtype=tf.float32)
  
  # 计算速度和压力的误差平方
  squared_error_uv = (prediction_velocity - inputs['target|velocity']) ** 2
  squared_error_p = (prediction_pressure - inputs['target|pressure']) ** 2

  # 计算累积均方误差 (MSE)
  cumulative_mse_uv = tf.cumsum(tf.reduce_mean(squared_error_uv, axis=[1, 2]), axis=0)
  cumulative_mse_p = tf.cumsum(tf.reduce_mean(squared_error_p, axis=[1, 2]))

  # 除以时间步长以得到平均累积 MSE
  average_cumulative_mse_uv = cumulative_mse_uv / time_steps
  average_cumulative_mse_p = cumulative_mse_p / time_steps
  
  # 计算 u 和 v 的相对均方根误差 (RMSE)
  cumulative_squared_error_u = tf.reduce_sum(squared_error_uv[:, :, 0], axis=1)
  cumulative_squared_error_v = tf.reduce_sum(squared_error_uv[:, :, 1], axis=1)

  denominator_u = tf.reduce_sum(prediction_velocity[:, :, 0] ** 2, axis=1)
  denominator_v = tf.reduce_sum(prediction_velocity[:, :, 1] ** 2, axis=1)

  cumulative_rmse_u = tf.cumsum(cumulative_squared_error_u / denominator_u, axis=0) / time_steps
  cumulative_rmse_v = tf.cumsum(cumulative_squared_error_v / denominator_v, axis=0) / time_steps

  # 计算 uv 和 p 的相对均方根误差 (RMSE)
  cumulative_squared_error_uv = tf.reduce_sum(squared_error_uv, axis=[1, 2])
  cumulative_squared_error_p = tf.reduce_sum(squared_error_p, axis=[1, 2])

  denominator_uv = tf.reduce_sum(prediction_velocity ** 2, axis=[1, 2])
  denominator_p = tf.reduce_sum(prediction_pressure ** 2, axis=[1, 2])

  cumulative_rmse_uv = tf.cumsum(cumulative_squared_error_uv / denominator_uv , axis=0) / time_steps
  cumulative_rmse_p = tf.cumsum(cumulative_squared_error_p / denominator_p , axis=0) / time_steps

  # 将累积误差转换为字典格式
  scalars = {}
  zero_start = tf.constant([0.0])
  
  # 使用 tf.concat 将 0.0 添加到每个累积误差序列的开始
  average_cumulative_mse_uv = tf.concat([zero_start, average_cumulative_mse_uv], axis=0)
  average_cumulative_mse_p = tf.concat([zero_start, average_cumulative_mse_p], axis=0)
  cumulative_rmse_u = tf.concat([zero_start, cumulative_rmse_u], axis=0)
  cumulative_rmse_v = tf.concat([zero_start, cumulative_rmse_v], axis=0)
  cumulative_rmse_uv = tf.concat([zero_start, cumulative_rmse_uv], axis=0)
  cumulative_rmse_p = tf.concat([zero_start, cumulative_rmse_p], axis=0)

  # 更新字典中的值
  scalars['uv_mse'] = average_cumulative_mse_uv
  scalars['p_mse'] = average_cumulative_mse_p
  scalars['u_rmse'] = cumulative_rmse_u
  scalars['v_rmse'] = cumulative_rmse_v
  scalars['uv_rmse'] = cumulative_rmse_uv
  scalars['p_rmse'] = cumulative_rmse_p

  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_velocity': inputs['target|velocity'],
      'gt_pressure': inputs['target|pressure'],
      'pred_velocity': prediction_velocity,
      'pred_pressure': prediction_pressure
  }
  return scalars, traj_ops
