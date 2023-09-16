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
  
  prediction_velocity = prediction[:,:,0:2]
  prediction_pressure = prediction[:,:,2:3]
  
  error_uv = tf.reduce_mean((prediction_velocity - inputs['velocity'])**2, axis=-1)
  
  scalars_uv_MSE = {'uv_mse_%d_steps' % horizon: tf.reduce_mean(error_uv[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200, 300, 400, 500, 599]}
  scalars_u = {'u_rmse_%d_steps' % horizon: tf.reduce_sum((prediction_velocity[1:horizon+1,:,0] - inputs['velocity'][1:horizon+1,:,0])**2)/tf.reduce_sum((prediction_velocity[1:horizon+1,:,0])**2) for horizon in [1, 10, 20, 50, 100, 200, 300, 400,599]}
  scalars_v = {'v_rmse_%d_steps' % horizon: tf.reduce_sum((prediction_velocity[1:horizon+1,:,1] - inputs['velocity'][1:horizon+1,:,1])**2)/tf.reduce_sum((prediction_velocity[1:horizon+1,:,1])**2) for horizon in [1, 10, 20, 50, 100, 200, 300, 400,599]}
  scalars_uv_RMSE = {'uv_rmse%d_steps' % horizon: tf.reduce_sum((prediction_velocity[1:horizon+1,:,:] - inputs['velocity'][1:horizon+1,:,:])**2)/tf.reduce_sum((prediction_velocity[1:horizon+1,:,:])**2) for horizon in [1, 10, 20, 50, 100, 200, 300, 400,599]}


  error_p = tf.reduce_mean((prediction_pressure - inputs['pressure'])**2, axis=-1)
  scalars_p_MSE = {'p_mse_%d_steps' % horizon: tf.reduce_mean(error_p[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200, 300, 400, 500, 599]}
  scalars_p_RMSE = {'p_rmse_%d_steps' % horizon: tf.reduce_sum((prediction_pressure[1:horizon+1,:,0] - inputs['pressure'][1:horizon+1,:,0])**2)/tf.reduce_sum((prediction_pressure[1:horizon+1,:,0])**2) for horizon in [1, 10, 20, 50, 100, 200, 300, 400,599]}

  scalars = scalars_uv_MSE
  
  for k,v in scalars_u.items():
    scalars[k]=v
    
  for k,v in scalars_v.items():
    scalars[k]=v
    
  for k,v in scalars_uv_RMSE.items():
    scalars[k]=v
    
  for k,v in scalars_p_MSE.items():
    scalars[k]=v
    
  for k,v in scalars_p_RMSE.items():
    scalars[k]=v 
    
  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_velocity': inputs['velocity'],
      'gt_pressure': inputs['pressure'],
      'pred_velocity': prediction_velocity,
      'pred_pressure': prediction_pressure
  }
  return scalars, traj_ops
