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

import csv
from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

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

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cfd', ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', '/lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/chk', 'Directory to save checkpoint')

'/home/doomduke/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/chk/origin_cylinder_chk/chk-full600steps'
'/home/doomduke/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/meshgraphnets/chk/new_hybrid_dataset'

flags.DEFINE_string('dataset_dir', '/lvm_data/litianyu/dataset/MeshGN/new_hybrid_dataset_Re=200-1500/origin', 'Directory to load dataset from.')

'/mnt/h/New-Hybrid-dataset/converted_dataset_bak/origin'
'/mnt/f/dataset/work1/train_dataset/tobias/original_dataset/cylinder_flow/cylinder_flow'

flags.DEFINE_string('rollout_path', '/lvm_data/litianyu/mycode-new/GEP-FVGN/repos-py/MeshGraphnets/offical-tf/meshgraphnets/chk/rollout.pkl',
                    'Pickle file to save eval trajectories')

flags.DEFINE_enum('rollout_split', 'test', ['train', 'test', 'valid','tri_wing_0_mesh_flow1_origin_0','generalization'],
                  'Dataset split to use for rollouts.')

flags.DEFINE_integer('num_rollouts', 100, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('batch_size', int(2), 'training batch size')
flags.DEFINE_bool('dual_edge', True, 'whether to encode undirect Graph')

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

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = False      #程序按需申请内存

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
  merged_summary_op = tf.summary.merge_all()
  
  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600,config=config) as sess:
    
    writer=tf.summary.FileWriter(f'{FLAGS.checkpoint_dir}/Logger', sess.graph)

    while not sess.should_stop():
      _, step, loss, scalar_data = sess.run([train_op, global_step, loss_op,merged_summary_op])
      writer.add_summary(scalar_data,step)
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
    
      scalars = {'uv_mse':[],
                 'p_mse':[],
                 'u_rmse':[],
                 'v_rmse':[],
                 'uv_rmse':[],
                 'p_rmse':[],
                 'relonyds_num':[]}
      
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
                                                                    uv_MSE=float(scalar_data['uv_mse'][-1]),
                                                                    plot_boundary_p_time_step=FLAGS.plot_boundary_p_time_step,
                                                                    save_tec=FLAGS.save_tec,
                                                                    plot_boundary=FLAGS.plot_boundary,
                                                                    traj_idx=traj_idx)
        
        scalar_data['relonyds_num'] = cell_center_trajectory['relonyds_num']
        
        '''plot mesh'''
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.cla()
        ax.clear()
        plt.gca().collections.clear()
        ax.set_aspect('equal')
        triang = mtri.Triangulation(cell_center_trajectory['mesh_pos'][0,:,0], cell_center_trajectory['mesh_pos'][0,:,1],cell_center_trajectory['cells_node'][0])
        bb_min = np.min(traj_data['gt_velocity'][0,:,0])
        bb_max = np.max(traj_data['gt_velocity'][0,:,0])
        cntr = ax.tripcolor(triang,traj_data['gt_velocity'][0,:,0], vmin=bb_min, vmax=bb_max)
        ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
        plt.colorbar(cntr)
        plt.savefig(saving_path+"/rollout_index"+str(traj_idx)+'velocity_field x-dir'+'.png')
        plt.close()
        '''plot mesh'''
    
        '''write error to file'''
        # 使用 f-string 完整构建文件路径
        file_name = f"Re({cell_center_trajectory['relonyds_num']:2e})_UV_RMSE({np.mean(scalar_data['uv_rmse']):2e}).csv"
        file_path = f"{saving_path}/{file_name}"
        
        # 使用 pickle 保存数据
        with open(file_path, 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)

          for key,value in scalar_data.items():
            
            # 写入标题
            writer.writerow([f'total_{key}'])
            # 写入数据行
            writer.writerow([np.mean(value)])
            
            scalars[key].append(scalar_data[key])
            
          # 写入标题
          writer.writerow(scalar_data.keys())
          # 写入数据行
          writer.writerow(scalar_data.values())
        '''>>> exit rollout loop >>>'''
        
      logging.info('-----ALL ROLLOUT DONE-----') 
      final_file_name = f"UV_RMSE({np.mean(np.stack(scalars['uv_rmse'], axis=0)):2e}).csv"
      final_file_path = f"{os.path.split(FLAGS.rollout_path)[0]}/{FLAGS.rollout_split}_{final_file_name}.csv"
      
      with open(final_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for key, value in scalars.items():
            if 'relonyds_num' in key:
              continue
            total_frames_error = np.stack(value, axis=0)
            
            logging.info('%s: %g', key, np.mean(total_frames_error))
            
            writer.writerow([f'total_{key}'])
            
            writer.writerow([np.mean(total_frames_error)])
              
            # 绘图
            plt.figure(figsize=(10, 5))  # 设置图形大小，这个大小需要根据实际需求调整
            mean_error = np.mean(total_frames_error, axis=0)
            scalars[key] = mean_error
            plt.plot(mean_error, label=key, color='red', linewidth=2)  # 指定线颜色和宽度

            # 设置标题和轴标签
            # plt.title('Error over Simulation Rollout Steps', fontsize=14)
            plt.xlabel('Simulation Rollout Step', fontsize=12)
            plt.ylabel('Relative-MSE', fontsize=12)

            # 设置图例
            plt.legend(loc='upper left', frameon=False)  # 关闭图例边框

            # 设置网格
            plt.grid(True, linestyle='--', alpha=0.5)  # 网格线样式

            # 设置轴的范围和刻度标签的字体大小
            plt.xlim(0, len(mean_error))  # 根据数据的长度设置 x 轴范围
            plt.ylim(0, np.max(mean_error) * 1.1)  # y 轴稍微高于最大误差值，留出一些空间
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # 显示并保存图形
            plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
            plt.show()
            
            # 保存图像
            output_path = os.path.join(os.path.split(FLAGS.rollout_path)[0], f"{FLAGS.rollout_split}_{key}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()  # 关闭当前窗口

          # 写入标题
        writer.writerow(scalars.keys())
        # 写入数据行
        writer.writerow(scalars.values())

def parse_node_center_to_cell_center(traj_data,rollout_index,uv_MSE,plot_boundary_p_time_step=None,save_tec=False,plot_boundary=False,traj_idx=None):
  
  traj_data['velocity'] = traj_data['gt_velocity']
  traj_data['pressure'] = traj_data['gt_pressure']
  cell_center_trajectory,cell_center_trajectory_with_boundary = extract_mesh_state(traj_data,None,rollout_index,mode="cylinder_mesh")
  rollout_time_length = cell_center_trajectory_with_boundary['target|UVP'].shape[0]
  
  saving_path = os.path.split(FLAGS.rollout_path)[0]+"/rollout_index"+str(rollout_index) +f'/Re(%2e)'%cell_center_trajectory['relonyds_num']+f'_mu(%e)'%FLAGS.mu+'_'+'rolloutsteps'+str(rollout_time_length)+f"_MSE(%2e)"%uv_MSE +'.dat'
  # if traj_idx==0:
  #   os.makedirs(os.path.split(saving_path)[0],exist_ok=True)
  #   sorted_keys = sorted(cell_center_trajectory_with_boundary.keys(), key=lambda x: x[0])
  #   sorted_center_trajectory_with_boundary = {k: cell_center_trajectory_with_boundary[k] for k in sorted_keys}
  #   with open(os.path.split(FLAGS.rollout_path)[0]+"/rollout_index"+str(rollout_index) +f'/Re(%2e)'%cell_center_trajectory['relonyds_num']+'data.json', 'w', encoding='utf-8') as f:
  #       converted_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sorted_center_trajectory_with_boundary.items()}
  #       json.dump(converted_dict, f, ensure_ascii=False, indent=4)
        
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
  params['batch'] = FLAGS.batch_size
  print(f"current running params: batch_size={params['batch']}, num_training_steps={FLAGS.num_training_steps}")
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15,
      dual_edge=FLAGS.dual_edge)
  model = params['model'].Model(learned_model,dual_edge=FLAGS.dual_edge)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
