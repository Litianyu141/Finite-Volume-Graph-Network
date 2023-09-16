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
"""Runs the learner/evaluator."""

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import json
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset
from meshgraphnets.parse_tfrecord_refactor import extract_mesh_state
from meshgraphnets.write_tec_plot_boundary  import write_tecplot_ascii_cell_centered
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'eval', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cfd', ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', '/home/litianyu/mycode/repos-py/MeshGraphnets/offical-tf/meshgraphnets/chk/cylinder_full_600steps/cylinder', 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', '/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset', 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', '/home/litianyu/mycode/repos-py/MeshGraphnets/offical-tf/meshgraphnets/rollout/cylinder_flow/full_600steps/rollout.pkl',
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'test', ['train', 'test', 'valid','tri_wing_0_mesh_flow1_origin_0'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 100, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')

flags.DEFINE_float('mu', float(0.001), 'viscosity of momentum')
flags.DEFINE_float('rho', float(1), 'density')

flags.DEFINE_bool('save_tec', False, 'whether saving tecplot file')
flags.DEFINE_bool('plot_boundary', True, 'whether saving plot boundary CL and CD')
flags.DEFINE_integer('plot_boundary_p_time_step', int(5e2), 'plot boundary pressure distubation at certain time step')
'''size is 3 , because we have output pressure'''
PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field=['velocity','pressure'], history=False,
                size=3, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存


def serialize_example(record):
    feature = {
        "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
        "cells": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells'].tobytes()])),
        "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
        "density": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['density'].tobytes()])),
        "pressure": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['pressure'].tobytes()])),
        "velocity": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['velocity'].tobytes()]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
def write_tfrecord_one(tfrecord_path,records):     
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
          serialized = serialize_example(records)
          writer.write(serialized)
          
def learner(model, params):
  """Run a learner job."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
  # print('proto shapes:', ds.output_shapes)
  # print('proto types:', ds.output_types)
  ds = dataset.add_targets(ds, params['field'], add_history=params['history'])
  ds = dataset.split_and_preprocess(ds, noise_field=params['field'][0],
                                    noise_scale=params['noise'],
                                    noise_gamma=params['gamma'])
  ds = dataset.batch_dataset(ds,params['batch'])
  # print('after splited shapes:', ds.output_shapes)
  # print('after splited types:', ds.output_types)
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  '''frames = {}
  save_tfrecord_path = '/mnt/h/repo-datasets/meshgn/airfoil/batches_dataset/batched-10-dataset.tfrecord'
  with tf.Session() as sess:
    for i in range(10):
        sliced_frames = sess.run(inputs)
        for key,val in sliced_frames.items():
            if i > 0:
              if key in ('density','pressure','velocity'):
                val = np.tile(val,(1,1,1))
                frames[key] = np.concatenate([frames[key],val],axis=0)
              else:
                pass
            else:
              if not(key=='target|velocity'):
                val = np.tile(val,(1,1,1))
                frames[key] = val  
              else:
                  pass
    write_tfrecord_one(save_tfrecord_path,frames)'''
  loss_op = model.loss(inputs)

  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(learning_rate=1e-4,
                                  global_step=global_step,
                                  decay_steps=int(5e6),
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)
  # Don't train for the first few steps, just accumulate normalization stats
  train_op = tf.cond(tf.less(global_step, 1000),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))
  # merged_summary_op = tf.summary.merge_all()
  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600,config=config) as sess:
    writer=tf.summary.FileWriter('chk/Logger', sess.graph)

    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      # writer.add_summary(scalar,step)
      if step % 1000 == 0:
        logging.info('Step %d: Loss %g', step, loss)
    writer.close()
    logging.info('Training complete.')


def evaluator(model, params):
  """Run a model rollout trajectory."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, params['field'], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None,
      config=config) as sess:
    with tf.device('/cpu:0'):
      trajectories = []
      scalars = []
      for traj_idx in range(FLAGS.num_rollouts):
        logging.info('Rollout trajectory %d', traj_idx)
        scalar_data, traj_data ,inputs_data= sess.run([scalar_op, traj_ops,inputs])
        traj_data['cells'] = inputs_data['cells']
        traj_data['node_type'] = inputs_data['node_type']
        
        '''make dir of current rollout for saving'''
        saving_path = os.path.split(FLAGS.rollout_path)[0]+"/rollout_index"+str(traj_idx)
        os.makedirs(saving_path,exist_ok=True)
        
        cell_center_trajectory,_ = parse_node_center_to_cell_center(traj_data=traj_data,
                                                                    rollout_index=traj_idx,
                                                                    uv_MSE=float(scalar_data['uv_mse_599_steps']),
                                                                    plot_boundary_p_time_step=FLAGS.plot_boundary_p_time_step,
                                                                    save_tec=FLAGS.save_tec,
                                                                    plot_boundary=FLAGS.plot_boundary)
        scalar_data['relonyds_num'] = cell_center_trajectory['relonyds_num']
        
        '''write error to file'''
        with open(saving_path+f'/Re(%2e)_'%cell_center_trajectory['relonyds_num']+f"MSE(%2e)_"%float(scalar_data['uv_mse_599_steps']) +'.txt','w') as f:
          f.write(f"-----rollout index {traj_idx}-----\n")
          for k,v in scalar_data.items():
            f.write(str(k)+' : '+str(v))
            f.write('\n')
            
        trajectories.append(traj_data)
        scalars.append(scalar_data)
      for key in scalars[0]:
        logging.info('-----ALL ROLLOUT DONE-----')
        logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
        
      with open(os.path.split(FLAGS.rollout_path)[0]+'/'+FLAGS.rollout_split+'.txt','w') as f:
        for index,scalar_data in enumerate(scalars):
          f.write(f"-----rollout index {index}-----\n")
          for k,v in scalar_data.items():
            f.write(str(k)+' : '+str(v))
            f.write('\n')
            
        f.write('-----ALL ROLLOUT DONE-----')
        f.write('%s: %g', key, np.mean([x[key] for x in scalars]))    
            
            
      # with open(FLAGS.rollout_path, 'wb') as fp:
      #   pickle.dump(trajectories, fp)

def parse_node_center_to_cell_center(traj_data,rollout_index,uv_MSE,plot_boundary_p_time_step=None,save_tec=False,plot_boundary=False):
  
  traj_data['velocity'] = traj_data['gt_velocity']
  traj_data['pressure'] = traj_data['gt_pressure']
  cell_center_trajectory,cell_center_trajectory_with_boundary = extract_mesh_state(traj_data,None,rollout_index,mode="cylinder_mesh")
  rollout_time_length = cell_center_trajectory_with_boundary['target|UVP'].shape[0]
  
  saving_path = os.path.split(FLAGS.rollout_path)[0]+"/rollout_index"+str(rollout_index) +f'/Re(%2e)'%cell_center_trajectory['relonyds_num']+f'_mu(%e)'%FLAGS.mu+'_'+'rolloutsteps'+str(rollout_time_length)+f"_MSE(%2e)"%uv_MSE +'.dat'
  os.makedirs(os.path.split(saving_path)[0],exist_ok=True)
  write_tecplot_ascii_cell_centered(raw_data=cell_center_trajectory_with_boundary,
                                    saving_path=saving_path,
                                    plot_boundary_p_time_step=plot_boundary_p_time_step,
                                    save_tec=save_tec,
                                    plot_boundary=plot_boundary
                                    )
  
  return cell_center_trajectory,cell_center_trajectory_with_boundary
  
  
  
def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[FLAGS.model]
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  model = params['model'].Model(learned_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
