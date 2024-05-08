#2 -*- encoding: utf-8 -*-
'''
@File    :   parse_tfrecord.py
@Author  :   litianyu 
@Version :   1.0
@Contact :   lty1040808318@163.com
'''
#解析tfrecord数据
import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py
import pickle
import enum
import multiprocessing as mp
import time
import sys
sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
import torch

import matplotlib.pyplot as plt
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation

from circle_fit import hyper_fit
c_NORMAL_max = 0
c_OBSTACLE_max = 0
c_AIRFOIL_max = 0
c_HANDLE_max = 0
c_INFLOW_max = 0
c_OUTFLOW_max = 0
c_WALL_BOUNDARY_max = 0
c_SIZE_max = 0

c_NORMAL_min = 3000
c_OBSTACLE_min = 3000
c_AIRFOIL_min = 3000
c_HANDLE_min = 3000
c_INFLOW_min  = 3000
c_OUTFLOW_min = 3000
c_WALL_BOUNDARY_min = 3000
c_SIZE_min = 3000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("done loading packges")
marker_cell_sp1 = []
marker_cell_sp2 = []

mask_of_mesh_pos_sp1 = np.arange(5233)
mask_of_mesh_pos_sp2 = np.arange(5233)

inverse_of_marker_cell_sp1 = np.arange(1)
inverse_of_marker_cell_sp2 = np.arange(1)

new_mesh_pos_iframe_sp1 = np.arange(1)
new_mesh_pos_iframe_sp2 = np.arange(1)

new_node_type_iframe_sp1 = np.arange(1)
new_node_type_iframe_sp2 = np.arange(1)

pos_dict_1 = {}
pos_dict_2 = {}
switch_of_redress = True

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
    REINFORCED_BOUNDARY = 10

case = 3
#980PRO
if case==0:
  path = {'tf_datasetPath':'H:/repo-datasets/meshgn/airfoil',
        'pickl_save_path':'H:/repo-datasets/meshgn/airfoil/pickle/airfoilonlyvsp1.pkl',
        'h5_save_path':'H:/repo-datasets/meshgn/h5/airfoil',
        'tfrecord_sp':'H://repo-datasets/meshgn/airfoil/tfrecord_splited/',
        'mesh_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'tec_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'saving_tec':False,
        'stastic':True,
        'saving_origin':True,
        'mask_features':True,
        'saving_sp_tf':True,
        'saving_sp_tf_single':False,
        'saving_sp_tf_mp':True,
        'saving_pickl': False,
        'saving_h5': False,
        'h5_sep' : False,
        'print_tf':False}
  model = {'name':'airfoil','maxepoch':10}
elif case==1:
#jtedu-lidequan1967
  path = {'tf_datasetPath':'/root/meshgraphnets/datasets/',
        'pickl_save_path':'/root/meshgraphnets/datasets/pickle/airfoilonlyvsp1.pkl',
        'h5_save_path':'/root/meshgraphnets/datasets/h5/',
        'tfrecord_sp':'/root/meshgraphnets/datasets/tfrecord_splited/',
        'mesh_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'tec_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'saving_tec':False,
        'stastic':True,
        'saving_origin':True,
        'mask_features':True,
        'saving_sp_tf':True,
        'saving_sp_tf_single':False,
        'saving_sp_tf_mp':True,
        'saving_pickl': False,
        'saving_h5': False,
        'h5_sep' : False,
        'print_tf':False}
  model = {'name':'airfoil','maxepoch':10}
elif case==2:
#jtedu-DOOMDUKE2
  path = {'tf_datasetPath':'/root/meshgraphnets/datasets/cylinder_flow/cylinder_flow/',
        'pickl_save_path':'/root/meshgraphnets/datasets/cylinder_flow/pickle/airfoil/cylinder_flowonlyvsp1.pkl',
        'h5_save_path':'/root/meshgraphnets/datasets/cylinder_flow/h5',
        'tfrecord_sp':'/root/meshgraphnets/datasets/cylinder_flow/tfrecord_splited/',
        'mesh_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'tec_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'saving_tec':False,
        'stastic':True,
        'saving_origin':True,
        'mask_features':False,
        'saving_sp_tf':True,
        'saving_sp_tf_single':False,
        'saving_sp_tf_mp':True,
        'saving_pickl': False,
        'saving_h5': False,
        'h5_sep' : False,
        'print_tf':False,
        'plot_order':True}
  model = {'name':'airfoil','maxepoch':10}
elif case==3:
#centre`s DL machine
  path = {'tf_datasetPath':'/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset',
        'pickl_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flowcylinder_flow.pkl',
        'h5_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow',
        'tfrecord_sp':'/data/litianyu/dataset/MeshGN/cylinder_flow/tfrecord_splited/',
        'mesh_save_path':'/home/litianyu/dataset/MeshGN/cylinder_flow/meshs_with_target_on_node_reinforced_boundary/',
        'tec_save_path':'/home/litianyu/mycode/repos-py/FVM/my_FVNN/rollouts/tecplot/',
        'mode':'cylinder_mesh',
        'renum_origin_dataset':False,
        'saving_tec':False,
        'stastic':False,
        'saving_origin':True,
        'mask_features':False,
        'saving_sp_tf':False,
        'saving_sp_tf_single':False,
        'saving_sp_tf_mp':False,
        'saving_pickl': False,
        'saving_h5': False,
        'h5_sep' : False,
        'print_tf':False,
        'plot_order':False}
  model = {'name':'aifoil','maxepoch':10}
elif case==4:
    #980-wsl-linux
  path = {'tf_datasetPath':'/mnt/h/repo-datasets/meshgn/airfoil',
        'pickl_save_path':'/mnt/h/repo-datasets/meshgn/airfoil/pickle/airfoilonlyvsp1.pkl',
        'h5_save_path':'/mnt/h/repo-datasets/meshgn/h5/airfoil',
        'tfrecord_sp':'/mnt/h/repo-datasets/meshgn/airfoil/tfrecord_splited/',
        'mesh_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'tec_save_path':'/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/',
        'saving_tec':False,
        'stastic':True,
        'saving_origin':True,
        'mask_features':True,
        'saving_sp_tf':True,
        'saving_sp_tf_single':False,
        'saving_sp_tf_mp':True,
        'saving_pickl': False,
        'saving_h5': False,
        'h5_sep' : False,
        'print_tf':False}
  model = {'name':'airfoil','maxepoch':10}
  
class parser:
  @staticmethod
  def mask_features(datasets,features:str,mask_factor):
    '''mask_factor belongs between [0,1]'''
    datasets['target|'+ features]=datasets[features]
    for frame_index in range(datasets[features].shape[0]):
      if features == 'velocity':  
        shape = datasets[features][frame_index].shape
        pre_mask = np.arange(shape[0])
        choosen_pos = np.random.choice(pre_mask,int(shape[0]*mask_factor))
        for i in range(choosen_pos.shape[0]):
          datasets[features][frame_index][choosen_pos[i]] = torch.zeros((1,shape[1]),dtype=torch.float32)  
        '''mask = np.random.randint(0, 2, size=shape[0])
        mask = np.expand_dims(mask,1).repeat(2,axis = 1)
        masked_velocity_frame = np.multiply(mask,datasets[features][frame_index])'''
    return datasets
def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  '''for index,frame in enumerate(ds):
      data = _parse(frame, meta)'''
  ds = ds.prefetch(1)
  return ds

def dividing_line(index,x):
      if index == 0:
        return x
      else:
        return 0.1*index*x
def stastic_nodeface_type(frame):
      flatten = frame[:,0]
      c_NORMAL = 0
      c_OBSTACLE = 0
      c_AIRFOIL = 0
      c_HANDLE = 0
      c_INFLOW = 0
      c_OUTFLOW = 0
      c_WALL_BOUNDARY = 0
      c_SIZE = 0
      for i in range(flatten.shape[0]):
            if(flatten[i]==NodeType.NORMAL):
                  c_NORMAL+=1
            elif(flatten[i]==NodeType.OBSTACLE):
                  c_OBSTACLE+=1
            elif(flatten[i]==NodeType.AIRFOIL):
                  c_AIRFOIL+=1
            elif(flatten[i]==NodeType.HANDLE):
                  c_HANDLE+=1
            elif(flatten[i]==NodeType.INFLOW):
                  c_INFLOW+=1
            elif(flatten[i]==NodeType.OUTFLOW):
                  c_OUTFLOW+=1
            elif(flatten[i]==NodeType.WALL_BOUNDARY):
                  c_WALL_BOUNDARY+=1
            elif(flatten[i]==NodeType.SIZE):
                  c_SIZE+=1
      print("NORMAL: {0} OBSTACLE: {1} AIRFOIL: {2} HANDLE: {3} INFLOW: {4} OUTFLOW: {5} WALL_BOUNDARY: {6} SIZE: {7}".format(c_NORMAL,c_OBSTACLE,c_AIRFOIL,c_HANDLE,c_INFLOW,c_OUTFLOW,c_WALL_BOUNDARY,c_SIZE))
def stastic(frame):
    flatten = frame[:,0]
    global c_NORMAL_max
    global c_OBSTACLE_max
    global c_AIRFOIL_max
    global c_HANDLE_max
    global c_INFLOW_max
    global c_OUTFLOW_max 
    global c_WALL_BOUNDARY_max 
    global c_SIZE_max 
    
    global c_NORMAL_min
    global c_OBSTACLE_min
    global c_AIRFOIL_min
    global c_HANDLE_min
    global c_INFLOW_min
    global c_OUTFLOW_min
    global c_WALL_BOUNDARY_min
    global c_SIZE_min
    
    c_NORMAL = 0
    c_OBSTACLE = 0
    c_AIRFOIL = 0
    c_HANDLE = 0
    c_INFLOW = 0
    c_OUTFLOW = 0
    c_WALL_BOUNDARY = 0
    c_SIZE = 0

      
    for i in range(flatten.shape[0]):
        if(flatten[i]==NodeType.NORMAL):
                c_NORMAL+=1

        elif(flatten[i]==NodeType.OBSTACLE):
                c_OBSTACLE+=1

        elif(flatten[i]==NodeType.AIRFOIL):
                c_AIRFOIL+=1

        elif(flatten[i]==NodeType.HANDLE):
                c_HANDLE+=1

        elif(flatten[i]==NodeType.INFLOW):
                c_INFLOW+=1

        elif(flatten[i]==NodeType.OUTFLOW):
                c_OUTFLOW+=1

        elif(flatten[i]==NodeType.WALL_BOUNDARY):
                c_WALL_BOUNDARY+=1

        elif(flatten[i]==NodeType.SIZE):
                c_SIZE+=1

    c_NORMAL_max = max(c_NORMAL_max,c_NORMAL)
    c_NORMAL_min = min(c_NORMAL_min,c_NORMAL)  
    c_OBSTACLE_max = max(c_OBSTACLE_max,c_OBSTACLE)
    c_OBSTACLE_min = min(c_OBSTACLE_min,c_OBSTACLE) 
    c_AIRFOIL_max = max(c_AIRFOIL_max,c_AIRFOIL)
    c_OBSTACLE_min = min(c_AIRFOIL_min,c_AIRFOIL)
    c_HANDLE_max = max(c_HANDLE_max,c_HANDLE)
    c_HANDLE_min = min(c_HANDLE_min,c_HANDLE)  
    c_INFLOW_max = max(c_INFLOW_max,c_INFLOW)
    c_INFLOW_min = min(c_INFLOW_min,c_INFLOW) 
    c_OUTFLOW_max = max(c_OUTFLOW_max,c_OUTFLOW)
    c_OUTFLOW_min = min(c_OUTFLOW_min,c_OUTFLOW)  
    c_WALL_BOUNDARY_max = max(c_WALL_BOUNDARY_max,c_WALL_BOUNDARY)
    c_WALL_BOUNDARY_min = min(c_WALL_BOUNDARY_min,c_WALL_BOUNDARY) 
    c_SIZE_max = max(c_SIZE_max,c_SIZE)
    c_SIZE_min = min(c_SIZE_min,c_SIZE)                                                                                          
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(record,mode='airfoil'):
  if mode=='airfoil':
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
  elif mode=='cylinder_flow':
    feature = {
        "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
        "cells": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells'].tobytes()])),
        "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
        "pressure": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['pressure'].tobytes()])),
        "velocity": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['velocity'].tobytes()]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
  elif mode == 'cylinder_mesh':
    feature = {
        "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
        "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
        "cells_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_node'].tobytes()])),
        "cell_factor": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cell_factor'].tobytes()])),
        "centroid": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['centroid'].tobytes()])),
        "face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face'].tobytes()])),
        "face_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_type'].tobytes()])),
        "face_length": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_length'].tobytes()])),
        "neighbour_cell": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['neighbour_cell'].tobytes()])),
        "cells_face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_face'].tobytes()])),
        "cells_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_type'].tobytes()])),
        "cells_area": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_area'].tobytes()])),
        "unit_norm_v": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['unit_norm_v'].tobytes()])),
        "target|velocity_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|velocity_on_node'].tobytes()])),
        "target|pressure_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|pressure_on_node'].tobytes()])),
        'mean_u': tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mean_u'].tobytes()])),
        'charac_scale': tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['charac_scale'].tobytes()]))
    }
  elif mode == 'airfoil_mesh':
    feature = {
        "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
        "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
        "cells_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_node'].tobytes()])),
        "cell_factor": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cell_factor'].tobytes()])),
        "centroid": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['centroid'].tobytes()])),
        "face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face'].tobytes()])),
        "face_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_type'].tobytes()])),
        "face_length": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_length'].tobytes()])),
        "neighbour_cell": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['neighbour_cell'].tobytes()])),
        "cells_face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_face'].tobytes()])),
        "cells_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_type'].tobytes()])),
        "cells_area": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_area'].tobytes()])),
        "unit_norm_v": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['unit_norm_v'].tobytes()])),
        "target|velocity_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|velocity_on_node'].tobytes()])),
        "target|pressure_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|pressure_on_node'].tobytes()]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
def write_tfrecord_one(tfrecord_path,records,mode):     
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
          serialized = serialize_example(records,mode=mode)
          writer.write(serialized)

def write_tfrecord_one_with_writer(writer,records,mode):     
          serialized = serialize_example(records,mode=mode)
          writer.write(serialized)

def write_tfrecord(tfrecord_path,records,np_index):     
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
      for index,frame in enumerate(records):
          serialized = serialize_example(frame)
          writer.write(serialized)
          print('process:{0} is writing traj:{1}'.format(np_index,index))
          

def write_tfrecord_mp(tfrecord_path_1,tfrecord_path_2,records):
    procs = []
    npc =0
    n_shard = 2
    for shard_id in range(n_shard):
        if(shard_id==0):
          args = (tfrecord_path_1, records[0],npc)
        elif(shard_id==1):
          args = (tfrecord_path_2, records[1],npc+1)
        p = mp.Process(target=write_tfrecord, args=args)
        p.start()
        procs.append(p)

    for proc in procs:
        proc.join()

def find_pos(mesh_point,mesh_pos_sp1):
    for k in range(mesh_pos_sp1.shape[0]):
        if (mesh_pos_sp1[k] == mesh_point).all():
            print('found{}'.format(k))
            return k
    return False
                              
def extract_mesh_state(dataset,writer,index,origin_writer=None,mode='cylinder_mesh'):
  INFLOW=0
  WALL_BOUNDARY=0
  OUTFLOW=0
  OBSTACLE = 0
  NORMAL=0

  mesh = {}
  mesh['mesh_pos'] = dataset['mesh_pos'][0]
  mesh['cells_node'] = np.sort(dataset['cells'][0] , axis=1)
  cells_node = torch.from_numpy(mesh['cells_node']).to(torch.int32)
  mesh['cells_node'] = np.expand_dims(cells_node,axis = 0)
  mesh_pos = mesh['mesh_pos']
  
  dataset['centroid'] = np.expand_dims((mesh_pos[mesh['cells_node'][0,:,0]]+mesh_pos[mesh['cells_node'][0,:,1]]+mesh_pos[mesh['cells_node'][0,:,2]])/3.,axis=0)
  
  mesh['centroid'] = dataset['centroid']
  
  # compose face        
  decomposed_cells = triangles_to_faces(cells_node,mesh['mesh_pos'])
  face = decomposed_cells['face_with_bias']
  senders = face[:,0]
  receivers = face[:,1]
  edge_with_bias = decomposed_cells['edge_with_bias']
  mesh['face'] = face.T.numpy().astype(np.int32)   
  
  # compute face length
  mesh['face_length'] = torch.norm(torch.from_numpy(mesh_pos)[senders] - torch.from_numpy(mesh_pos)[receivers], dim=-1,keepdim=True).to(torch.float32).numpy()
  
  # check-out face_type
  face_type = np.zeros((mesh['face'].shape[1],1),dtype=np.int32)
  a = torch.index_select(torch.from_numpy(dataset['node_type'][0]),0,torch.from_numpy(mesh['face'][0])).numpy()
  b = torch.index_select(torch.from_numpy(dataset['node_type'][0]),0,torch.from_numpy(mesh['face'][1])).numpy()
  face_center_pos = (torch.index_select(torch.from_numpy(mesh['mesh_pos']),0,torch.from_numpy(mesh['face'][0])).numpy()+torch.index_select(torch.from_numpy(mesh['mesh_pos']),0,torch.from_numpy(mesh['face'][1])).numpy())/2.
  
  if mode.find('airfoil')!=-1:
    face_type = torch.from_numpy(face_type)
    Airfoil = torch.full(face_type.shape,NodeType.AIRFOIL).to(torch.int32)
    Interior = torch.full(face_type.shape,NodeType.NORMAL).to(torch.int32)
    Inlet = torch.full(face_type.shape,NodeType.INFLOW).to(torch.int32)
    a = torch.from_numpy(a).view(-1)
    b = torch.from_numpy(b).view(-1)
    face_type[(a==b)&(a==NodeType.AIRFOIL)&(b==NodeType.AIRFOIL),:] = Airfoil[(a==b)&(a==NodeType.AIRFOIL)&(b==NodeType.AIRFOIL),:]
    face_type[(a==b)&(a==NodeType.NORMAL)&(b==NodeType.NORMAL),:] = Interior[(a==b)&(a==NodeType.NORMAL)&(b==NodeType.NORMAL),:]
    face_type[(a==b)&(a==NodeType.INFLOW)&(b==NodeType.INFLOW),:] = Inlet[(a==b)&(a==NodeType.INFLOW)&(b==NodeType.INFLOW),:]
    
  else:
    for i in range(face_center_pos.shape[0]):
        topwall = np.max(face_center_pos[:,1])
        bottomwall = np.min(face_center_pos[:,1])
        outlet = np.max(face_center_pos[:,0])
        inlet = np.min(face_center_pos[:,0])
        current_coord = face_center_pos[i]
        if current_coord[0] == inlet:
            face_type[i] = NodeType.INFLOW
            INFLOW+=1
        elif current_coord[1] == topwall or current_coord[1] == bottomwall:
            face_type[i] = NodeType.WALL_BOUNDARY
            WALL_BOUNDARY+=1
        elif current_coord[0] == outlet:   
            face_type[i] = NodeType.OUTFLOW 
            OUTFLOW+=1
        elif (a[i]>0) and (b[i]>0) and current_coord[0]>0 and current_coord[0]<outlet and current_coord[1]>0 and current_coord[1]<topwall:
            face_type[i] = NodeType.WALL_BOUNDARY 
            WALL_BOUNDARY+=1
            OBSTACLE+=1
        else:
            face_type[i] = NodeType.NORMAL 
            NORMAL+=1
  mesh['face_type'] = face_type
  # print("After renumed data has:")
  # stastic_nodeface_type(face_type)
                  
  # compute cell_face index and cells_type
  face_list = torch.from_numpy(mesh['face']).transpose(0,1).numpy()
  face_index = {}
  for i in range(face_list.shape[0]):
      face_index[str(face_list[i])] = i              
  nodes_of_cell = torch.stack(torch.chunk(edge_with_bias,3,0),dim=1)

  nodes_of_cell = nodes_of_cell.numpy()
  edges_of_cell = np.ones((nodes_of_cell.shape[0],nodes_of_cell.shape[1]),dtype=np.int32)
  cells_type = np.zeros((nodes_of_cell.shape[0],1),dtype=np.int32)
  for i in range(nodes_of_cell.shape[0]):
    three_face_index = [face_index[str(nodes_of_cell[i][0])],face_index[str(nodes_of_cell[i][1])],face_index[str(nodes_of_cell[i][2])]]
    three_face_type = [face_type[three_face_index[0]],face_type[three_face_index[1]],face_type[three_face_index[2]]]
    INFLOW_t = 0
    WALL_BOUNDARY_t = 0
    OUTFLOW_t = 0
    AIRFOIL_t = 0
    NORMAL_t = 0
    for type in three_face_type:
      if type == NodeType.INFLOW:
        INFLOW_t+=1
      elif type == NodeType.WALL_BOUNDARY:
        WALL_BOUNDARY_t+=1
      elif type == NodeType.OUTFLOW:
        OUTFLOW_t+=1
      elif type == NodeType.AIRFOIL:
        AIRFOIL_t+=1
      else:
        NORMAL_t+=1
    if INFLOW_t>0 and WALL_BOUNDARY_t>0 and NORMAL_t>0:
      cells_type[i] = NodeType.INFLOW
    elif OUTFLOW_t>0 and WALL_BOUNDARY_t>0 and NORMAL_t>0:
      cells_type[i] = NodeType.OUTFLOW
    elif WALL_BOUNDARY_t>0 and NORMAL_t>0 and INFLOW_t==0 and OUTFLOW_t==0:
      cells_type[i] = NodeType.WALL_BOUNDARY
    elif AIRFOIL_t>0 and NORMAL_t>0 and INFLOW_t==0 and OUTFLOW_t==0:
      cells_type[i] = NodeType.AIRFOIL
    elif INFLOW_t>0 and NORMAL_t>0 and WALL_BOUNDARY_t==0 and OUTFLOW_t==0:
      cells_type[i] = NodeType.INFLOW
    elif OUTFLOW_t>0 and NORMAL_t>0 and WALL_BOUNDARY_t==0 and INFLOW_t==0:
      cells_type[i] = NodeType.OUTFLOW
    else:
      cells_type[i] = NodeType.NORMAL
    for j in range(3):
      single_face_index = face_index[str(nodes_of_cell[i][j])]
      edges_of_cell[i][j] = single_face_index
  mesh['cells_face'] = edges_of_cell
  mesh['cells_type'] = cells_type
  # print("cells_type:")
  # stastic_nodeface_type(cells_type)
  # mesh_pos = mesh['centroid'][0]
  # node_type = cells_type.reshape(-1)
  # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
  # ax.cla()
  # ax.set_aspect('equal')
  # #triang = mtri.Triangulation(display_pos[:, 0], display_pos[:, 1])
  # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
  # #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
  # plt.scatter(mesh_pos[node_type==NodeType.NORMAL,0],mesh_pos[node_type==NodeType.NORMAL,1],c='red',linewidths=1)
  # plt.scatter(mesh_pos[node_type==NodeType.AIRFOIL,0],mesh_pos[node_type==NodeType.AIRFOIL,1],c='green',linewidths=1)
  # plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='blue',linewidths=1)
  # plt.show()
  
  # unit normal vector
  pos_diff = torch.index_select(torch.from_numpy(mesh['mesh_pos']),0,senders)-torch.index_select(torch.from_numpy(mesh['mesh_pos']),0,receivers)
  unv = torch.cat((-pos_diff[:,1:2],pos_diff[:,0:1]),dim=1)
  for i in range(unv.shape[0]):
    if torch.isinf(unv[i][1]):
      unv[i] = torch.tensor([0,1],dtype=torch.float32)
  unv = unv/torch.norm(unv,dim=1).view(-1,1)

  #TODO:complete the normal vector calculation
  face_center_pos = (torch.index_select(torch.from_numpy(mesh['mesh_pos']),0,torch.from_numpy(mesh['face'][0])).numpy()+torch.index_select(torch.from_numpy(mesh['mesh_pos']),0,torch.from_numpy(mesh['face'][1])).numpy())/2.
  centroid = torch.from_numpy(mesh['centroid'][0])
  cells_face = torch.from_numpy(mesh['cells_face']).T
  
   # calc cell version of unv, and make sure all unv point outside of the cell 
  edge_unv_set = []
  for i in range(3):
    edge_1 = cells_face[i]
    edge_1_uv =  torch.index_select(unv,0,edge_1)
    edge_1_center_centroid_vec = torch.from_numpy(mesh['centroid'][0])-torch.index_select(torch.from_numpy(face_center_pos),0,edge_1)
    edge_uv_times_ccv = edge_1_uv[:,0]*edge_1_center_centroid_vec[:,0]+edge_1_uv[:,1]*edge_1_center_centroid_vec[:,1]
    Edge_op = torch.logical_or(edge_uv_times_ccv>0,torch.full(edge_uv_times_ccv.shape,False))
    Edge_op = torch.stack((Edge_op,Edge_op),dim=-1)
    edge_1_unv = torch.where(Edge_op,edge_1_uv*(-1.),edge_1_uv)
    edge_unv_set.append(edge_1_unv.unsqueeze(1))
  mesh['unit_norm_v'] = torch.cat(edge_unv_set,dim=1).numpy()
  
  #compute face`s neighbor cell index
  cell_dict = {}
  edge_index  = np.expand_dims(mesh['face'],axis=0)

  count_1 = 0
  for i in range(nodes_of_cell.shape[0]):
      edge_1 = str(nodes_of_cell[i, 0])
      edge_2 = str(nodes_of_cell[i, 1])
      edge_3 = str(nodes_of_cell[i, 2])

      if edge_1 in cell_dict:
          cell_dict[edge_1]=[cell_dict[edge_1][0],np.asarray(i, dtype=np.int32)]
          count_1+=1
      else:
          cell_dict[edge_1] = [np.asarray(i, dtype=np.int32)]

      if edge_2 in cell_dict:
          cell_dict[edge_2]=[cell_dict[edge_2][0],np.asarray(i, dtype=np.int32)]
          count_1+=1
      else:
          cell_dict[edge_2] = [np.asarray(i, dtype=np.int32)]
          
      if edge_3 in cell_dict:
          cell_dict[edge_3]=[cell_dict[edge_3][0],np.asarray(i, dtype=np.int32)]
          count_1+=1
      else:
          cell_dict[edge_3] = [np.asarray(i, dtype=np.int32)]
  edge_index_t = edge_index.transpose(0,2,1)
  neighbour_cell = np.zeros_like(edge_index_t)
  face_node_index = edge_index_t
  count = 0
  for i in range(edge_index_t.shape[1]):
      face_str = str(face_node_index[0, i, :])
      cell_index = cell_dict[face_str]
      if len(cell_index) > 1:
          neighbour_cell[0][i] = np.stack((cell_index[0],cell_index[1]),axis=0)
      else:
          neighbour_cell[0][i] = np.stack((cell_index[0],cell_index[0]), axis=0) #adding self-loop instead of ghost cell
          count+=1
  # plot_edge_direction(centroid,torch.from_numpy(neighbour_cell[0]))        
  neighbour_cell_with_bias = reorder_face(centroid,torch.from_numpy(neighbour_cell[0]),plot=False)
  mesh['neighbour_cell'] = neighbour_cell_with_bias.unsqueeze(0).numpy().transpose(0,2,1)

  #compute cell attribute V_BIC and P_BIC and 
  node_index_of_cell = cells_node.transpose(1,0)
  v_target = dataset['velocity']
  p_target = dataset['pressure']
  cell_node_dist = torch.sum((torch.index_select(torch.from_numpy(dataset['mesh_pos']),1,node_index_of_cell[0])[0:1,:,:]-centroid.view(1,-1,2))**2,dim=2).view(1,-1,1)+torch.sum((torch.index_select(torch.from_numpy(dataset['mesh_pos']),1,node_index_of_cell[1])[0:1,:,:]-centroid.view(1,-1,2))**2,dim=2).view(1,-1,1)+torch.sum((torch.index_select(torch.from_numpy(dataset['mesh_pos']),1,node_index_of_cell[2])[0:1,:,:]-centroid.view(1,-1,2))**2,dim=2).view(1,-1,1)
  cell_factor_1 = torch.sum((torch.index_select(torch.from_numpy(dataset['mesh_pos']),1,node_index_of_cell[0])[0:1,:,:]-centroid.view(1,-1,2))**2,dim=2).view(1,-1,1)/cell_node_dist
  cell_factor_2 = torch.sum((torch.index_select(torch.from_numpy(dataset['mesh_pos']),1,node_index_of_cell[1])[0:1,:,:]-centroid.view(1,-1,2))**2,dim=2).view(1,-1,1)/cell_node_dist
  cell_factor_3 = torch.sum((torch.index_select(torch.from_numpy(dataset['mesh_pos']),1,node_index_of_cell[2])[0:1,:,:]-centroid.view(1,-1,2))**2,dim=2).view(1,-1,1)/cell_node_dist
  mesh['cell_factor'] = torch.cat((cell_factor_1,cell_factor_2,cell_factor_3),dim=2).numpy()
  mesh['target|velocity_on_node'] = v_target # obviously, velocity with BC, IC is v_target[0]
  mesh['target|pressure_on_node'] = p_target # obviously, velocity with BC, IC is v_pressure[1]
  mesh['predicted|velocity_on_node'] = dataset['pred_velocity']
  mesh['predicted|pressure_on_node'] = dataset['pred_pressure']
  if mode.find('airfoil') != -1:
    mesh['target|density'] = dataset['density']
  
  #compute cell_area
  cells_face = torch.from_numpy(mesh['cells_face']).T
  face_length = torch.from_numpy(mesh['face_length'])
  len_edge_1 = torch.index_select(face_length,0,cells_face[0])
  len_edge_2 = torch.index_select(face_length,0,cells_face[1])
  len_edge_3 = torch.index_select(face_length,0,cells_face[2])
  p = (1./2.)*(len_edge_1+len_edge_2+len_edge_3)
  cells_area = torch.sqrt(p*(p-len_edge_1)*(p-len_edge_2)*(p-len_edge_3))
  mesh['cells_area'] = cells_area.numpy()
  
  #edge attr
  mesh['node_type'] = np.expand_dims(dataset['node_type'][0],axis=0)
  mesh['mesh_pos'] = np.expand_dims(mesh['mesh_pos'],axis=0)
  mesh['face'] = np.expand_dims(mesh['face'],axis=0)
  mesh['face_type'] = np.expand_dims(mesh['face_type'],axis=0)
  mesh['face_length'] = np.expand_dims(mesh['face_length'],axis=0)
  mesh['cells_face'] = np.expand_dims(mesh['cells_face'],axis=0)
  mesh['cells_type'] = np.expand_dims(mesh['cells_type'],axis=0)
  mesh['cells_area'] = np.expand_dims(mesh['cells_area'],axis=0)
  mesh['unit_norm_v'] = np.expand_dims(mesh['unit_norm_v'],axis=0)


  print('{0}th mesh has been extracted'.format(index))
  trajectory = cal_relonyds_number(trajectory = mesh, mu=0.001, rho=1.)
  cylinder_node_mask, cylinder_face_mask , cylinder_cell_mask = extract_cylinder_boundary(trajectory)
  
  trajectory_with_boundary = transform_to_cell_center_traj(trajectory,cylinder_node_mask, cylinder_face_mask , cylinder_cell_mask, index)
  
  return trajectory,trajectory_with_boundary

def transform_to_cell_center_traj(trajectory,cylinder_node_mask,cylinder_face_mask,cylinder_cell_mask,rollout_index):
  
  result_to_tec_and_plot = {}
  traj_length = trajectory['target|velocity_on_node'].shape[0]  
  target_velocity_on_node = torch.from_numpy(trajectory['target|velocity_on_node'])
  target_pressure_on_node = torch.from_numpy(trajectory['target|pressure_on_node'])
  target_uvp_on_node = torch.cat((target_velocity_on_node,target_pressure_on_node),dim=2)
  
  predicted_velocity_on_node = torch.from_numpy(trajectory['predicted|velocity_on_node'])
  predicted_pressure_on_node = torch.from_numpy(trajectory['predicted|pressure_on_node'])
  predicted_uvp_on_node = torch.cat((predicted_velocity_on_node,predicted_pressure_on_node),dim=2)
  
  cell_factor = torch.from_numpy(trajectory['cell_factor'][0])
  cell_node = torch.from_numpy(trajectory['cells_node'][0]).T
  face_node = torch.from_numpy(trajectory['face'][0])
  
  target_uvp_on_cell = cell_factor[:,0:1]*(torch.index_select(target_uvp_on_node,1,cell_node[0]))+cell_factor[:,1:2]*(torch.index_select(target_uvp_on_node,1,cell_node[1]))+cell_factor[:,2:3]*(torch.index_select(target_uvp_on_node,1,cell_node[2]))
  
  target_uvp_on_edge = (torch.index_select(target_uvp_on_node,1,face_node[0])+torch.index_select(target_uvp_on_node,1,face_node[1]))/2.
  
  predicted_uvp_on_cell = cell_factor[:,0:1]*(torch.index_select(predicted_uvp_on_node,1,cell_node[0]))+cell_factor[:,1:2]*(torch.index_select(predicted_uvp_on_node,1,cell_node[1]))+cell_factor[:,2:3]*(torch.index_select(predicted_uvp_on_node,1,cell_node[2]))
  
  predicted_uvp_on_edge = (torch.index_select(predicted_uvp_on_node,1,face_node[0])+torch.index_select(predicted_uvp_on_node,1,face_node[1]))/2.

  result_to_tec_and_plot['mean_u'] = trajectory['mean_u']
  result_to_tec_and_plot['reynolds_num'] = trajectory['reynolds_num']
  result_to_tec_and_plot['mesh_pos'] = trajectory['mesh_pos'].repeat(traj_length,axis=0)
  result_to_tec_and_plot['cells'] = trajectory['cells_node'].repeat(traj_length,axis=0)
  result_to_tec_and_plot['node_type'] = trajectory['node_type'].repeat(traj_length,axis=0)
  
  # only UVP are cell centered variables
  result_to_tec_and_plot['cell_type']= trajectory['cells_type'].repeat(traj_length,axis=0)
  result_to_tec_and_plot['velocity'] = predicted_uvp_on_cell[:,:,0:2].numpy()
  result_to_tec_and_plot['pressure'] = predicted_uvp_on_cell[:,:,2:3].numpy()
  result_to_tec_and_plot['face_length'] = trajectory['face_length'].repeat(traj_length,axis=0)

  result_to_tec_and_plot['boundary_mesh_pos'] = torch.from_numpy(trajectory['mesh_pos'][:,cylinder_node_mask,:]).numpy().repeat(traj_length,axis=0)
  result_to_tec_and_plot['boundary_pressure'] = predicted_uvp_on_edge[:,cylinder_face_mask,2:3].numpy()
  result_to_tec_and_plot['boundary_velocity'] = predicted_uvp_on_edge[:,cylinder_face_mask,0:2].numpy()

  result_to_tec_and_plot['cylinder_boundary_cells'] = torch.from_numpy(trajectory['cells_node'])[:,cylinder_cell_mask,:].transpose(1,2).numpy().repeat(traj_length,axis=0)
  result_to_tec_and_plot['cylinder_boundary_cell_unv'] = torch.from_numpy(trajectory['unit_norm_v'])[:,cylinder_cell_mask,:,:].numpy().repeat(traj_length,axis=0)
  result_to_tec_and_plot['cylinder_boundary_cell_area'] = torch.from_numpy(trajectory['cells_area'])[:,cylinder_cell_mask,:].numpy().repeat(traj_length,axis=0)
  result_to_tec_and_plot['cylinder_boundary_cell_face'] = torch.from_numpy(trajectory['cells_face'])[:,cylinder_cell_mask,:].transpose(1,2).numpy().repeat(traj_length,axis=0)
  result_to_tec_and_plot['predicted_edge_UVP'] = predicted_uvp_on_edge.numpy()
  result_to_tec_and_plot['target_edge_UVP'] = target_uvp_on_edge.numpy()
  result_to_tec_and_plot['target_cylinder|pressure'] = target_uvp_on_edge[:,cylinder_face_mask,2:3].numpy()
  result_to_tec_and_plot['target_cylinder|velocity'] = target_uvp_on_edge[:,cylinder_face_mask,0:2].numpy()
  result_to_tec_and_plot['boundary_face'] = torch.from_numpy(trajectory['face']).transpose(1,2)[:,cylinder_face_mask,:].numpy().repeat(traj_length,axis=0)
  result_to_tec_and_plot['centroid'] = torch.from_numpy(trajectory['centroid'])[0,:,:].numpy()
  result_to_tec_and_plot['cell_area'] = torch.from_numpy(trajectory['cells_area'])[0,:,:].numpy()
  result_to_tec_and_plot['target|UVP'] = target_uvp_on_cell.numpy()
  
  print(f'transformed graph index{rollout_index} with boundary data')
  
  return result_to_tec_and_plot
  
  
def cal_relonyds_number(trajectory,mu,rho):
    
    '''prepare data for cal_relonyds_number'''
    target_on_node = torch.cat((torch.from_numpy(trajectory['target|velocity_on_node'][0]),torch.from_numpy(trajectory['target|pressure_on_node'][0])),dim=1)
    edge_index = torch.from_numpy(trajectory['face'][0])
    target_on_edge = (torch.index_select(target_on_node,0,edge_index[0])+torch.index_select(target_on_node,0,edge_index[1]))/2.
    face_type = torch.from_numpy(trajectory['face_type'][0]).view(-1)
    node_type = torch.from_numpy(trajectory['node_type'][0]).view(-1)
    Inlet = target_on_edge[face_type==NodeType.INFLOW][:,0]
    face_length = torch.from_numpy(trajectory['face_length'])[0][:,0][face_type==NodeType.INFLOW]
    total_u = torch.sum(Inlet*face_length)
    mesh_pos = torch.from_numpy(trajectory['mesh_pos'][0])
    top = torch.max(mesh_pos[:,1]).numpy()
    bottom = torch.min(mesh_pos[:,1]).numpy()
    left = torch.min(mesh_pos[:,0]).numpy()
    right = torch.max(mesh_pos[:,0]).numpy()
    mean_u = total_u/(top-bottom)
    
    '''cal cylinder diameter'''
    boundary_pos = mesh_pos[node_type==NodeType.WALL_BOUNDARY].numpy()
    cylinder_mask = torch.full((boundary_pos.shape[0],1),True).view(-1).numpy()
    cylinder_not_mask = np.logical_not(cylinder_mask)
    cylinder_mask = np.where(((boundary_pos[:,1]>bottom)&(boundary_pos[:,1]<top)&(boundary_pos[:,0]<right)&(boundary_pos[:,0]>left)),cylinder_mask,cylinder_not_mask)
    cylinder_pos = torch.from_numpy(boundary_pos[cylinder_mask])
    _,_,R,_= hyper_fit(np.asarray(cylinder_pos))
    L0 = R*2.
    rho = rho
    mu = mu
    trajectory['reynolds_num'] = ((mean_u*L0*rho)/mu).numpy()
    trajectory['mean_u'] = mean_u.numpy()
    return trajectory 
  
def extract_cylinder_boundary(mesh):
    
    face_node = torch.from_numpy(mesh['face'][0])
    node_type = torch.from_numpy(mesh['node_type'][0])
    mesh_pos = torch.from_numpy(mesh['mesh_pos'][0])
    centroid = torch.from_numpy(mesh['centroid'][0])
    cell_face = torch.from_numpy(mesh['cells_face'][0]).T
    cell_node = torch.from_numpy(mesh['cells_node'][0]).T
    cell_type = torch.from_numpy(mesh['cells_type'][0])
    cells_three_pos = [torch.index_select(mesh_pos,0,cell_node[0]),torch.index_select(mesh_pos,0,cell_node[1]),torch.index_select(mesh_pos,0,cell_node[2])]
    
    node_topwall = torch.max(mesh_pos[:,1])
    node_bottomwall = torch.min(mesh_pos[:,1])
    node_outlet = torch.max(mesh_pos[:,0])
    node_inlet = torch.min(mesh_pos[:,0])
    
    face_type = torch.from_numpy(mesh['face_type'][0])
    left_face_node_pos = torch.index_select(mesh_pos,0,face_node[0])
    right_face_node_pos = torch.index_select(mesh_pos,0,face_node[1])  
     
    left_face_node_type = torch.index_select(node_type,0,face_node[0])
    right_face_node_type = torch.index_select(node_type,0,face_node[1])  
                      
    face_center_pos = (left_face_node_pos+right_face_node_pos)/2.
    
    face_topwall = torch.max(face_center_pos[:,1])
    face_bottomwall = torch.min(face_center_pos[:,1])
    face_outlet = torch.max(face_center_pos[:,0])
    face_inlet = torch.min(face_center_pos[:,0])
    
    centroid_topwall = torch.max(centroid[:,1])
    centroid_bottomwall = torch.min(centroid[:,1])
    centroid_outlet = torch.max(centroid[:,0])
    centroid_inlet = torch.min(centroid[:,0])
    
    MasknodeT = torch.full((mesh_pos.shape[0],1),True)
    MasknodeF = torch.logical_not(MasknodeT)

    MaskfaceT = torch.full((face_node.shape[1],1),True)
    MaskfaceF = torch.logical_not(MaskfaceT)
    
    MaskcellT = torch.full((cell_face.shape[1],1),True)
    MaskcellF = torch.logical_not(MaskcellT)
    
    cylinder_node_mask = torch.where(((node_type==NodeType.WALL_BOUNDARY)&(mesh_pos[:,1:2]<node_topwall)&(mesh_pos[:,1:2]>node_bottomwall)&(mesh_pos[:,0:1]>node_inlet)&(mesh_pos[:,0:1]<node_outlet)),MasknodeT,MasknodeF).squeeze(1)
    
    cylinder_face_mask = torch.where(((face_type==NodeType.WALL_BOUNDARY)&(face_center_pos[:,1:2]<face_topwall)&(face_center_pos[:,1:2]>face_bottomwall)&(face_center_pos[:,0:1]>face_inlet)&(face_center_pos[:,0:1]<face_outlet)&(left_face_node_pos[:,1:2]<node_topwall)&(left_face_node_pos[:,1:2]>node_bottomwall)&(left_face_node_pos[:,0:1]>node_inlet)&(left_face_node_pos[:,0:1]<node_outlet)&(right_face_node_pos[:,1:2]<node_topwall)&(right_face_node_pos[:,1:2]>node_bottomwall)&(right_face_node_pos[:,0:1]>node_inlet)&(right_face_node_pos[:,0:1]<node_outlet)&(left_face_node_type==NodeType.WALL_BOUNDARY)&(right_face_node_type==NodeType.WALL_BOUNDARY)),MaskfaceT,MaskfaceF).squeeze(1)
    
    cylinder_cell_mask = torch.where(((cell_type==NodeType.WALL_BOUNDARY)&(centroid[:,1:2]<centroid_topwall)&(centroid[:,1:2]>centroid_bottomwall)&(centroid[:,0:1]>centroid_inlet)&(centroid[:,0:1]<centroid_outlet)&(cells_three_pos[0][:,1:2]<centroid_topwall)&(cells_three_pos[0][:,1:2]>centroid_bottomwall)&(cells_three_pos[0][:,0:1]>centroid_inlet)&(cells_three_pos[0][:,0:1]<centroid_outlet)&(cells_three_pos[1][:,1:2]<centroid_topwall)&(cells_three_pos[1][:,1:2]>centroid_bottomwall)&(cells_three_pos[1][:,0:1]>centroid_inlet)&(cells_three_pos[1][:,0:1]<centroid_outlet)&(cells_three_pos[2][:,1:2]<centroid_topwall)&(cells_three_pos[2][:,1:2]>centroid_bottomwall)&(cells_three_pos[2][:,0:1]>centroid_inlet)&(cells_three_pos[2][:,0:1]<centroid_outlet)),MaskcellT,MaskcellF).squeeze(1)
    
    #plt.scatter(face_center_pos[cylinder_face_mask].cpu().numpy()[:,0],face_center_pos[cylinder_face_mask].cpu().numpy()[:,1],edgecolors='red')
    #plt.show()
    return cylinder_node_mask, cylinder_face_mask , cylinder_cell_mask
  
def make_dim_less(trajectory,params=None):
    target_on_node = torch.cat((torch.from_numpy(trajectory['target|velocity_on_node'][0]),torch.from_numpy(trajectory['target|pressure_on_node'][0])),dim=1)
    edge_index = torch.from_numpy(trajectory['face'][0])
    target_on_edge = (torch.index_select(target_on_node,0,edge_index[0])+torch.index_select(target_on_node,0,edge_index[1]))/2.
    face_type = torch.from_numpy(trajectory['face_type'][0]).view(-1)
    node_type = torch.from_numpy(trajectory['node_type'][0]).view(-1)
    Inlet = target_on_edge[face_type==NodeType.INFLOW][:,0]
    face_length = torch.from_numpy(trajectory['face_length'])[0][:,0][face_type==NodeType.INFLOW]
    total_u = torch.sum(Inlet*face_length)
    mesh_pos = torch.from_numpy(trajectory['mesh_pos'][0])
    top = torch.max(mesh_pos[:,1]).numpy()
    bottom = torch.min(mesh_pos[:,1]).numpy()
    mean_u = total_u/(top-bottom)
    
    boundary_pos = mesh_pos[node_type==NodeType.WALL_BOUNDARY].numpy()
    cylinder_mask = torch.full((boundary_pos.shape[0],1),True).view(-1).numpy()
    cylinder_not_mask = np.logical_not(cylinder_mask)
    cylinder_mask = np.where(((boundary_pos[:,1]>bottom)&(boundary_pos[:,1]<top)),cylinder_mask,cylinder_not_mask)
    
    cylinder_pos = torch.from_numpy(boundary_pos[cylinder_mask])
    
    xc,yc,R,_= hyper_fit(np.asarray(cylinder_pos))
    
    # R = torch.norm(cylinder_pos[0]-torch.tensor([xc,yc]))
    L0 = R*2.
    
    rho = 1
    
    trajectory['target|velocity_on_node'] = ((trajectory['target|velocity_on_node'])/mean_u).numpy()
    
    trajectory['target|pressure_on_node'] = (trajectory['target|pressure_on_node']/((mean_u**2)*(L0**2)*rho)).numpy()
    
    trajectory['mean_u'] = mean_u.view(1,1,1).numpy()
    trajectory['charac_scale'] = L0.view(1,1,1).numpy()
    return trajectory
      
def seprate_cells(mesh_pos,cells,node_type,density,pressure,velocity,index):
  global marker_cell_sp1
  global marker_cell_sp2
  global mask_of_mesh_pos_sp1
  global mask_of_mesh_pos_sp2
  global inverse_of_marker_cell_sp1
  global inverse_of_marker_cell_sp2
  global pos_dict_1
  global pos_dict_2
  global switch_of_redress
  global new_node_type_iframe_sp1
  global new_node_type_iframe_sp2
  global new_mesh_pos_iframe_sp1
  global new_mesh_pos_iframe_sp2

  marker_cell_sp1 = []
  marker_cell_sp2 = []

  #separate cells into two species and saved as marker cells
  for i in range(cells.shape[1]):
    cell = cells[0][i]
    cell = cell.tolist()
    member = 0
    for j in cell:
          x_cord = mesh_pos[0][j][0]
          y_cord = mesh_pos[0][j][1]
          if dividing_line(index,x_cord)>=0:
                marker_cell_sp1.append(cell)
                member+=1
                break
    if(member == 0):
      marker_cell_sp2.append(cell)
  marker_cell_sp1 = np.asarray(marker_cell_sp1,dtype = np.int32)
  marker_cell_sp2 = np.asarray(marker_cell_sp2,dtype = np.int32)
  
  #use mask to filter the mesh_pos of sp1 and sp2
  marker_cell_sp1_flat = marker_cell_sp1.reshape((marker_cell_sp1.shape[0])*(marker_cell_sp1.shape[1]))
  marker_cell_sp1_flat_uq = np.unique(marker_cell_sp1_flat)

  marker_cell_sp2_flat = marker_cell_sp2.reshape((marker_cell_sp2.shape[0])*(marker_cell_sp2.shape[1]))
  marker_cell_sp2_flat_uq = np.unique(marker_cell_sp2_flat)
  
  #mask filter of mesh_pos tensor
  inverse_of_marker_cell_sp1 = np.delete(mask_of_mesh_pos_sp1,marker_cell_sp1_flat_uq)
  inverse_of_marker_cell_sp2 = np.delete(mask_of_mesh_pos_sp2,marker_cell_sp2_flat_uq)
  
  #apply mask for mesh_pos first
  new_mesh_pos_iframe_sp1 = np.delete(mesh_pos[0],inverse_of_marker_cell_sp1,axis=0)
  new_mesh_pos_iframe_sp2 = np.delete(mesh_pos[0],inverse_of_marker_cell_sp2,axis=0)
  
  #redress the mesh_pos`s indexs in the marker_cells,because,np.delete would only delete the element charged by the index and reduce the length of splited mesh_pos_frame tensor,but the original mark_cells stores the original index of the mesh_pos tensor,so we need to redress the indexs
  count = 0
  for i in range(new_mesh_pos_iframe_sp1.shape[0]):
      pos_dict_1[str(new_mesh_pos_iframe_sp1[i])] = i
  
  for index in range(marker_cell_sp1.shape[0]):
        cell = marker_cell_sp1[index]
        for j in range(3):
            mesh_point = str(mesh_pos[0][cell[j]])
            if(pos_dict_1.get(mesh_point,10000) < 6000):
                  marker_cell_sp1[index][j] = pos_dict_1[mesh_point]
            if (marker_cell_sp1[index][j]>new_mesh_pos_iframe_sp1.shape[0]):
                  count+=1
                  print("有{0}个cell 里的 meshPoint的索引值超过了meshnodelist的长度".format(count))
              
  for r in range(new_mesh_pos_iframe_sp2.shape[0]):
      pos_dict_2[str(new_mesh_pos_iframe_sp2[r])] = r
      
  count_1 = 0
  for index in range(marker_cell_sp2.shape[0]):
        cell = marker_cell_sp2[index]
        for j in range(3):
            mesh_point = str(mesh_pos[0][cell[j]])
            if(pos_dict_2.get(mesh_point,10000) < 6000):
                    marker_cell_sp2[index][j] = pos_dict_2[mesh_point]
            if (marker_cell_sp2[index][j]>new_mesh_pos_iframe_sp2.shape[0]):
                  count_1+=1
                  print("有{0}个cell 里的 meshPoint的索引值超过了meshnodelist的长度".format(count))
                  
  new_node_type_iframe_sp1 = np.delete(node_type[0],inverse_of_marker_cell_sp1,axis=0)
  new_node_type_iframe_sp2 = np.delete(node_type[0],inverse_of_marker_cell_sp2,axis=0)
    
  new_velocity_sp1 = np.delete(velocity,inverse_of_marker_cell_sp1,axis=1)
  new_density_sp1 = np.delete(density,inverse_of_marker_cell_sp1,axis=1)
  new_pressure_sp1 = np.delete(pressure,inverse_of_marker_cell_sp1,axis=1)
  
  new_velocity_sp2 = np.delete(velocity,inverse_of_marker_cell_sp2,axis=1)
  new_density_sp2 = np.delete(density,inverse_of_marker_cell_sp2,axis=1)
  new_pressure_sp2 = np.delete(pressure,inverse_of_marker_cell_sp2,axis=1)
      
  #rearrange_frame with v_sp1 and reshape to 1 dim
  start_time = time.time()
  re_mesh_pos = np.asarray([new_mesh_pos_iframe_sp1])  #mesh is static, so store only once will be enough not 601
  re_cells = np.tile(marker_cell_sp1,(1,1,1))  #same as above
  re_node_type = np.asarray([new_node_type_iframe_sp1])  #same as above
  rearrange_frame_1['node_type'] = re_node_type
  rearrange_frame_1['cells'] = re_cells
  rearrange_frame_1['mesh_pos'] = re_mesh_pos
  rearrange_frame_1['density'] = new_density_sp1
  rearrange_frame_1['pressure'] = new_pressure_sp1
  rearrange_frame_1['velocity'] = new_velocity_sp1
  
  re_mesh_pos = np.asarray([new_mesh_pos_iframe_sp2])  #mesh is static, so store only once will be enough not 601
  re_cells = np.tile(marker_cell_sp2,(1,1,1))  #same as above
  re_node_type = np.asarray([new_node_type_iframe_sp2])  #same as above
  rearrange_frame_2['node_type'] = re_node_type
  rearrange_frame_2['cells'] = re_cells
  rearrange_frame_2['mesh_pos'] = re_mesh_pos
  rearrange_frame_2['density'] = new_density_sp2
  rearrange_frame_2['pressure'] = new_pressure_sp2
  rearrange_frame_2['velocity'] = new_velocity_sp2
  
  end_time = time.time()
  
  return [rearrange_frame_1,rearrange_frame_2]

def transform_2(mesh_pos,cells,node_type,density,pressure,velocity,index):
  seprate_cells_result = seprate_cells(cells,mesh_pos)
  new_mesh_pos_sp1 = []
  new_mesh_pos_sp2 = []
  marker_cell_sp1 = np.asarray(seprate_cells_result[0],dtype = np.int32)
  marker_cell_sp2 = np.asarray(seprate_cells_result[0],dtype = np.int32)
  
  for i in range(marker_cell_sp1.shape[0]):
    cell =  marker_cell_sp1[0][i]
    mesh_pos[0]

def parse_reshape(ds):
  rearrange_frame = {}
  re_mesh_pos= np.arange(1)
  re_node_type = np.arange(1)
  re_velocity = np.arange(1)
  re_cells = np.arange(1)
  re_density =  np.arange(1)
  re_pressure = np.arange(1)
  count = 0
  for index, d in enumerate(ds):
      if count==0:
        re_mesh_pos = np.expand_dims(d['mesh_pos'].numpy(),axis=0)
        re_node_type = np.expand_dims(d['node_type'].numpy(),axis=0)
        re_velocity = np.expand_dims(d['velocity'].numpy(),axis=0)
        re_cells = np.expand_dims(d['cells'].numpy(),axis=0)
        re_density =  np.expand_dims(d['density'].numpy(),axis=0)
        re_pressure = np.expand_dims(d['pressure'].numpy(),axis=0)
        count+=1
        print("No.{0} has been added to the dict".format(index))
      else:
        re_mesh_pos = np.insert(re_mesh_pos,index,d['mesh_pos'].numpy(),axis=0)
        re_node_type = np.insert(re_node_type,index,d['node_type'].numpy(),axis=0)
        re_velocity = np.insert(re_velocity,index,d['velocity'].numpy(),axis=0)
        re_cells = np.insert(re_cells,index,d['cells'].numpy(),axis=0)
        re_density = np.insert(re_density,index,d['density'].numpy(),axis=0)
        re_pressure = np.insert(re_pressure,index,d['pressure'].numpy(),axis=0)
        print("No.{0} has been added to the dict".format(index))
  rearrange_frame['node_type'] = re_node_type
  rearrange_frame['cells'] = re_cells
  rearrange_frame['mesh_pos'] = re_mesh_pos
  rearrange_frame['density'] = re_density
  rearrange_frame['pressure'] = re_pressure
  rearrange_frame['velocity'] = re_velocity
  print('done')
  return rearrange_frame      
def reorder_face(mesh_pos,edges,plot=False):
  
    senders = edges[:,0]
    receivers = edges[:,1]
    
    edge_vec = torch.index_select(mesh_pos,0,senders)-torch.index_select(mesh_pos,0,receivers)
    e_x = torch.cat((torch.ones(edge_vec.shape[0],1), (torch.zeros(edge_vec.shape[0],1))),dim=1)
    
    edge_vec_dot_ex = edge_vec[:,0]*e_x[:,0]+edge_vec[:,1]*e_x[:,1]

    edge_op = torch.logical_or(edge_vec_dot_ex>0,torch.full(edge_vec_dot_ex.shape,False))
    edge_op = torch.stack((edge_op,edge_op),dim=-1)
    
    edge_op_1 = torch.logical_and(edge_vec[:,0]==0,edge_vec[:,1]>0)
    edge_op_1 = torch.stack((edge_op_1,edge_op_1),dim=-1)
    
    unique_edges = torch.stack((senders,receivers), dim=1)
    inverse_unique_edges = torch.stack((receivers,senders), dim=1)
    
    edge_with_bias = torch.where(((edge_op)|(edge_op_1)),unique_edges,inverse_unique_edges)

    if plot:
      plot_edge_direction(mesh_pos,edge_with_bias)
    
    return edge_with_bias
  
def plot_edge_direction(mesh_pos,edges):
  
    senders = edges[:,0]
    receivers = edges[:,1]
    
    edge_vec = torch.index_select(mesh_pos,0,senders)-torch.index_select(mesh_pos,0,receivers)
    e_x = torch.cat((torch.ones(edge_vec.shape[0],1), (torch.zeros(edge_vec.shape[0],1))),dim=1)
    e_y = torch.cat((torch.zeros(edge_vec.shape[0],1), (torch.ones(edge_vec.shape[0],1))),dim=1)
    
    edge_vec_dot_ex = edge_vec[:,0]*e_x[:,0]+edge_vec[:,1]*e_x[:,1]
    edge_vec_dot_ey = edge_vec[:,0]*e_y[:,0]+edge_vec[:,1]*e_y[:,1]
    
    cosu = edge_vec_dot_ex/((torch.norm(edge_vec,dim=1)*torch.norm(e_x,dim=1)))
    cosv = edge_vec_dot_ey/((torch.norm(edge_vec,dim=1)*torch.norm(e_y,dim=1)))
    
    plt.quiver(torch.index_select(mesh_pos[:,0:1],0,senders),torch.index_select(mesh_pos[:,1:2],0,senders),edge_vec[:,0],edge_vec[:,1],units='height',scale=1.2,width=0.0025)
    
    plt.show()

  
  
def triangles_to_faces(faces, mesh_pos,deform=False):
    """Computes mesh edges from triangles."""
    mesh_pos = torch.from_numpy(mesh_pos)
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        unique_edges = torch.stack((senders,receivers), dim=1)
        
        #plot_edge_direction(mesh_pos,unique_edges)
        
        face_with_bias = reorder_face(mesh_pos,unique_edges,plot=False)
        edge_with_bias = reorder_face(mesh_pos,packed_edges,plot=False)
        return {'two_way_connectivity': two_way_connectivity, 
                'senders': senders, 
                'receivers': receivers,
                'unique_edges':unique_edges,
                'face_with_bias':face_with_bias,
                'edge_with_bias':edge_with_bias}
      
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
      
#This function is compromised to Tobias Paffs`s datasets
def mask_face_bonudary(face_types,faces,velocity_on_node,pressure_on_node,is_train=False):
  
  if is_train:
    
    velocity_on_face = ((torch.index_select(velocity_on_node,0,faces[0])+torch.index_select(velocity_on_node,0,faces[1]))/2.).numpy()
    pressure_on_face = ((torch.index_select(pressure_on_node,0,faces[0])+torch.index_select(pressure_on_node,0,faces[1]))/2.).numpy()
    
  else:
    
    velocity_on_face = ((torch.index_select(torch.from_numpy(velocity_on_node),1,torch.from_numpy(faces[0]))+torch.index_select(torch.from_numpy(velocity_on_node),1,torch.from_numpy(faces[1])))/2.)
    pressure_on_face = ((torch.index_select(torch.from_numpy(pressure_on_node),1,torch.from_numpy(faces[0]))+torch.index_select(torch.from_numpy(pressure_on_node),1,torch.from_numpy(faces[1])))/2.)
    '''
    face_types = torch.from_numpy(face_types)
    mask_of_p = torch.zeros_like(pressure_on_face)
    mask_of_v = torch.zeros_like(velocity_on_face)
    pressure_on_face_t = torch.where((face_types==NodeType.OUTFLOW)|(face_types==NodeType.INFLOW),pressure_on_face,mask_of_p)
    face_types = face_types.repeat(1,1)
    velocity_on_face_t = torch.where((face_types==NodeType.OUTFLOW)|(face_types==NodeType.INFLOW),velocity_on_face,mask_of_v).repeat(1,3)
    '''
  return torch.cat((velocity_on_face,pressure_on_face),dim=2).numpy()
def direction_bias(dataset):
  mesh_pos = dataset['mesh_pos'][0]
  edge_vec = dataset['face']
def renum_data(dataset,unorder=True,index=0,plot='cell'):
  re_index = np.linspace(0,int(dataset['mesh_pos'].shape[1])-1,int(dataset['mesh_pos'].shape[1])).astype(np.int64)
  re_cell_index = np.linspace(0,int(dataset['cells'].shape[1])-1,int(dataset['cells'].shape[1])).astype(np.int64)
  key_list = []
  new_dataset = {}
  for key,value in dataset.items():
    dataset[key] = torch.from_numpy(value)
    key_list.append(key)
    
    
  new_dataset = {}
  cells_node = dataset['cells'][0]
  dataset['centroid'] = np.zeros((cells_node.shape[0],2),dtype = np.float32)
  for index_c in range(cells_node.shape[0]):
          cell = cells_node[index_c]
          centroid_x = 0.0
          centroid_y = 0.0
          for j in range(3):
              centroid_x += dataset['mesh_pos'].numpy()[0][cell[j]][0]
              centroid_y += dataset['mesh_pos'].numpy()[0][cell[j]][1]
          dataset['centroid'][index_c] = np.array([centroid_x/3,centroid_y/3],dtype=np.float32)
  dataset['centroid'] = torch.from_numpy(np.expand_dims(dataset['centroid'],axis = 0))

  if unorder:
    np.random.shuffle(re_index)
    np.random.shuffle(re_cell_index)
    for key in key_list:
      value = dataset[key]
      if key=='cells':
        # TODO: cells_node is not correct, need implementation
        new_dataset[key]=torch.index_select(value,1,torch.from_numpy(re_cell_index).to(torch.long))
      elif  key=='boundary':
        new_dataset[key]=value
      else:
        new_dataset[key] = torch.index_select(value,1,torch.from_numpy(re_index).to(torch.long))
    cell_renum_dict = {}
    new_cells = np.empty_like(dataset['cells'][0])
    for i in range(new_dataset['mesh_pos'].shape[1]):
      cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i
      
    for j in range(dataset['cells'].shape[1]):
      cell = new_dataset['cells'][0][j]
      for node_index in range(cell.shape[0]):
        new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
    new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 ) 
    new_dataset['cells'] = torch.from_numpy(new_cells)
    
    cells_node = new_dataset['cells'][0]
    mesh_pos = new_dataset['mesh_pos']
    new_dataset['centroid'] = ((torch.index_select(mesh_pos,1,cells_node[:,0])+torch.index_select(mesh_pos,1,cells_node[:,1])+torch.index_select(mesh_pos,1,cells_node[:,2]))/3.).view(1,-1,2)
    for key,value in new_dataset.items():
      dataset[key] = value.numpy()
      new_dataset[key] = value.numpy()
  
  else:
    
    data_cell_centroid = dataset['centroid'].to(torch.float64)[0]
    data_cell_Z = -8*data_cell_centroid[:,0]**(2)+3*data_cell_centroid[:,0]*data_cell_centroid[:,1]-2*data_cell_centroid[:,1]**(2)+20.
    data_node_pos = dataset['mesh_pos'].to(torch.float64)[0]
    data_Z = -8*data_node_pos[:,0]**(2)+3*data_node_pos[:,0]*data_node_pos[:,1]-2*data_node_pos[:,1]**(2)+20.
    a = np.unique(data_Z.cpu().numpy(), return_counts=True)
    b = np.unique(data_cell_Z.cpu().numpy(), return_counts=True)
    if a[0].shape[0] !=data_Z.shape[0] or b[0].shape[0] !=data_cell_Z.shape[0]:
      print("data renum faild{0}".format(index))
      return False,False
    else:
      sorted_data_Z,new_data_index = torch.sort(data_Z,descending=False)
      sorted_data_cell_Z,new_data_cell_index = torch.sort(data_cell_Z,descending=False)
      for key in key_list:
        value = dataset[key]
        if key=='cells':
          new_dataset[key]=torch.index_select(value,1,new_data_cell_index)
        elif key=='boundary':
          new_dataset[key]=value
        else:
          new_dataset[key] = torch.index_select(value,1,new_data_index)
      cell_renum_dict = {}
      new_cells = np.empty_like(dataset['cells'][0])
      for i in range(new_dataset['mesh_pos'].shape[1]):
        cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i
      for j in range(dataset['cells'].shape[1]):
        cell = dataset['cells'][0][j]
        for node_index in range(cell.shape[0]):
          new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
      new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 ) 
      new_dataset['cells'] = torch.index_select(torch.from_numpy(new_cells),1,new_data_cell_index)

      cells_node = new_dataset['cells'][0]
      mesh_pos = new_dataset['mesh_pos'][0]
      new_dataset['centroid'] = ((torch.index_select(mesh_pos,0,cells_node[:,0])+torch.index_select(mesh_pos,0,cells_node[:,1])+torch.index_select(mesh_pos,0,cells_node[:,2]))/3.).view(1,-1,2)
      for key,value in new_dataset.items():
        dataset[key] = value.numpy()
        new_dataset[key] = value.numpy()
      #new_dataset = reorder_boundaryu_to_front(dataset)  
  if plot is not None and plot=='cell':
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.cla()
    ax.set_aspect('equal')
    #bb_min = mesh['velocity'].min(axis=(0, 1))
    #bb_max = mesh['velocity'].max(axis=(0, 1))
    mesh_pos = new_dataset['mesh_pos'][0]
    faces = new_dataset['cells'][0]
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    plt.show()
  elif plot is not None and plot=='node':
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.cla()
    ax.set_aspect('equal')
    #bb_min = mesh['velocity'].min(axis=(0, 1))
    #bb_max = mesh['velocity'].max(axis=(0, 1))
    mesh_pos = new_dataset['mesh_pos'][0]
    #faces = dataset['cells'][0]
    #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    plt.show()
    
  elif plot is not None and plot=='centroid':
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.cla()
    ax.set_aspect('equal')
    #bb_min = mesh['velocity'].min(axis=(0, 1))
    #bb_max = mesh['velocity'].max(axis=(0, 1))
    mesh_pos = new_dataset['centroid'][0]
    #faces = dataset['cells'][0]
    #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    plt.show()
    
  elif plot is not None and plot=='plot_order':
    fig = plt.figure()  # 创建画布
    mesh_pos = new_dataset['mesh_pos'][0]
    centroid = new_dataset['centroid'][0]
    display_centroid_list=[centroid[0],centroid[1],centroid[2]]
    display_pos_list=[mesh_pos[0],mesh_pos[1],mesh_pos[2]]
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    def animate(num):
      
      if num < mesh_pos.shape[0]:
        display_pos_list.append(mesh_pos[num])
      display_centroid_list.append(centroid[num])
      if num%3 ==0 and num >0:
          display_pos = np.array(display_pos_list)
          display_centroid = np.array(display_centroid_list)
          p1 = ax1.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
          ax1.legend(['node_pos'], loc=2, fontsize=10)
          p2 = ax2.scatter(display_centroid[:,0],display_centroid[:,1],c='green',linewidths=1)
          ax2.legend(['centriod'], loc=2, fontsize=10)
    ani = animation.FuncAnimation(fig, animate, frames=new_dataset['centroid'][0].shape[0], interval=100)    
    if unorder:         
      ani.save("unorder"+"test.gif", writer='pillow') 
    else:         
      ani.save("order"+"test.gif", writer='pillow')      
  return new_dataset,True
def reorder_boundaryu_to_front(dataset,plot=None):
  
  boundary_attributes = {}
  
  node_type = torch.from_numpy(dataset['node_type'][0])[:,0]
  face_type = torch.from_numpy(dataset['face_type'][0])[:,0]
  cells_type = torch.from_numpy(dataset['cells_type'][0])[:,0]
  
  node_mask_t = torch.full(node_type.shape,True)
  node_mask_i = torch.logical_not(node_mask_t)
  face_mask_t = torch.full(face_type.shape,True)
  face_mask_i = torch.logical_not(face_mask_t)
  cells_mask_t = torch.full(cells_type.shape,True)
  cells_mask_i = torch.logical_not(cells_mask_t)
  
  node_mask = torch.where(node_type==NodeType.NORMAL,node_mask_t,node_mask_i)
  face_mask = torch.where(face_type==NodeType.NORMAL,face_mask_t,face_mask_i)
  cells_mask = torch.where(cells_type==NodeType.NORMAL,cells_mask_t,cells_mask_i)
  
  
  boundary_node_mask = torch.logical_not(node_mask)
  boundary_face_mask = torch.logical_not(face_mask)
  boundary_cells_mask = torch.logical_not(cells_mask)
  
  '''boundary attributes'''
  for key,value in dataset.items():
    if key == 'mesh_pos':
      boundary_attributes = value[:,boundary_node_mask,:]
      Interior_attributes = value[:,node_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'target|velocity_on_node':
      boundary_attributes = value[:,boundary_node_mask,:]
      Interior_attributes = value[:,node_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'target|pressure_on_node':
      boundary_attributes = value[:,boundary_node_mask,:]
      Interior_attributes = value[:,node_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'node_type':
      boundary_attributes = value[:,boundary_node_mask,:]
      Interior_attributes = value[:,node_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'cells_node':
      boundary_attributes = value[:,boundary_cells_mask,:]
      Interior_attributes = value[:,cells_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'centroid':
      boundary_attributes = value[:,boundary_cells_mask,:]
      Interior_attributes = value[:,cells_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'face':
      boundary_attributes = value[:,:,boundary_face_mask]
      Interior_attributes = value[:,:,face_mask]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=2)
    elif key == 'face_length':
      boundary_attributes = value[:,boundary_face_mask,:]
      Interior_attributes = value[:,face_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'face_type':
      boundary_attributes = value[:,boundary_face_mask,:]
      Interior_attributes = value[:,face_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'cells_face':
      boundary_attributes = value[:,boundary_cells_mask,:]
      Interior_attributes = value[:,cells_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'cells_type':
      boundary_attributes = value[:,boundary_cells_mask,:]
      Interior_attributes = value[:,cells_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'unit_norm_v':
      boundary_attributes = value[:,boundary_cells_mask,:,:]
      Interior_attributes = value[:,cells_mask,:,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'neighbour_cell':
      boundary_attributes = value[:,:,boundary_face_mask]
      Interior_attributes = value[:,:,face_mask]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=2)
    elif key == 'cell_factor':
      boundary_attributes = value[:,boundary_cells_mask,:]
      Interior_attributes = value[:,cells_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
    elif key == 'cells_area':
      boundary_attributes = value[:,boundary_cells_mask,:]
      Interior_attributes = value[:,cells_mask,:]
      dataset[key] = np.concatenate((boundary_attributes,Interior_attributes),axis=1)
        
  if plot is not None and plot=='cell':
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.cla()
    ax.set_aspect('equal')
    #bb_min = mesh['velocity'].min(axis=(0, 1))
    #bb_max = mesh['velocity'].max(axis=(0, 1))
    mesh_pos = dataset['mesh_pos'][0]
    faces = dataset['cells'][0]
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    plt.show()
  elif plot is not None and plot=='node':
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.cla()
    ax.set_aspect('equal')
    #bb_min = mesh['velocity'].min(axis=(0, 1))
    #bb_max = mesh['velocity'].max(axis=(0, 1))
    mesh_pos = dataset['mesh_pos'][0]
    #faces = dataset['cells'][0]
    #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    plt.show()
    
  elif plot is not None and plot=='centroid':
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.cla()
    ax.set_aspect('equal')
    #bb_min = mesh['velocity'].min(axis=(0, 1))
    #bb_max = mesh['velocity'].max(axis=(0, 1))
    mesh_pos = dataset['centroid'][0]
    #faces = dataset['cells'][0]
    #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    plt.show()
    
  elif plot is not None and plot=='plot_order':
    fig = plt.figure()  # 创建画布
    mesh_pos = dataset['mesh_pos'][0]
    centroid = dataset['centroid'][0]
    display_centroid_list=[centroid[0],centroid[1],centroid[2]]
    display_pos_list=[mesh_pos[0],mesh_pos[1],mesh_pos[2]]
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    def animate(num):
      
      if num < mesh_pos.shape[0]:
        display_pos_list.append(mesh_pos[num])
      display_centroid_list.append(centroid[num])
      if num%3 ==0 and num >0:
          display_pos = np.array(display_pos_list)
          display_centroid = np.array(display_centroid_list)
          p1 = ax1.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
          ax1.legend(['node_pos'], loc=2, fontsize=10)
          p2 = ax2.scatter(display_centroid[:,0],display_centroid[:,1],c='green',linewidths=1)
          ax2.legend(['centriod'], loc=2, fontsize=10)
    ani = animation.FuncAnimation(fig, animate, frames=dataset['centroid'][0].shape[0], interval=100) 
    plt.show(block=True)  
    ani.save("order"+"test.gif", writer='pillow')      
  return dataset
def parse_origin_dataset(dataset,unorder=False,index_num=0,plot=None,writer=None):
  re_index = np.linspace(0,int(dataset['mesh_pos'].shape[1])-1,int(dataset['mesh_pos'].shape[1])).astype(np.int64)
  re_cell_index = np.linspace(0,int(dataset['cells'].shape[1])-1,int(dataset['cells'].shape[1])).astype(np.int64)
  key_list = []
  new_dataset = {}
  for key,value in dataset.items():
    dataset[key] = torch.from_numpy(value)
    key_list.append(key)
    
  new_dataset = {}
  cells_node = dataset['cells'][0]
  dataset['centroid'] = np.zeros((cells_node.shape[0],2),dtype = np.float32)
  for index_c in range(cells_node.shape[0]):
          cell = cells_node[index_c]
          centroid_x = 0.0
          centroid_y = 0.0
          for j in range(3):
              centroid_x += dataset['mesh_pos'].numpy()[0][cell[j]][0]
              centroid_y += dataset['mesh_pos'].numpy()[0][cell[j]][1]
          dataset['centroid'][index_c] = np.array([centroid_x/3,centroid_y/3],dtype=np.float32)
  dataset['centroid'] = torch.from_numpy(np.expand_dims(dataset['centroid'],axis = 0))

  if unorder:
    np.random.shuffle(re_index)
    np.random.shuffle(re_cell_index)
    for key in key_list:
      value = dataset[key]
      if key=='cells':
        # TODO: cells_node is not correct, need implementation
        new_dataset[key]=torch.index_select(value,1,torch.from_numpy(re_cell_index).to(torch.long))
      elif  key=='boundary':
        new_dataset[key]=value
      else:
        new_dataset[key] = torch.index_select(value,1,torch.from_numpy(re_index).to(torch.long))
    cell_renum_dict = {}
    new_cells = np.empty_like(dataset['cells'][0])
    for i in range(new_dataset['mesh_pos'].shape[1]):
      cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i
      
    for j in range(dataset['cells'].shape[1]):
      cell = new_dataset['cells'][0][j]
      for node_index in range(cell.shape[0]):
        new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
    new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 ) 
    new_dataset['cells'] = torch.from_numpy(new_cells)
    
    cells_node = new_dataset['cells'][0]
    mesh_pos = new_dataset['mesh_pos']
    new_dataset['centroid'] = ((torch.index_select(mesh_pos,1,cells_node[:,0])+torch.index_select(mesh_pos,1,cells_node[:,1])+torch.index_select(mesh_pos,1,cells_node[:,2]))/3.).view(1,-1,2)
    for key,value in new_dataset.items():
      dataset[key] = value.numpy()
      new_dataset[key] = value.numpy()
  
  else:
    
    data_cell_centroid = dataset['centroid'].to(torch.float64)[0]
    data_cell_Z = -8*data_cell_centroid[:,0]**(2)+3*data_cell_centroid[:,0]*data_cell_centroid[:,1]-2*data_cell_centroid[:,1]**(2)+20.
    data_node_pos = dataset['mesh_pos'].to(torch.float64)[0]
    data_Z = -8*data_node_pos[:,0]**(2)+3*data_node_pos[:,0]*data_node_pos[:,1]-2*data_node_pos[:,1]**(2)+20.
    a = np.unique(data_Z.cpu().numpy(), return_counts=True)
    b = np.unique(data_cell_Z.cpu().numpy(), return_counts=True)
    if a[0].shape[0] !=data_Z.shape[0] or b[0].shape[0] !=data_cell_Z.shape[0]:
      print("data renum faild{0}".format(index))
      return False
    else:
      sorted_data_Z,new_data_index = torch.sort(data_Z,descending=False)
      sorted_data_cell_Z,new_data_cell_index = torch.sort(data_cell_Z,descending=False)
      for key in key_list:
        value = dataset[key]
        if key=='cells':
          new_dataset[key]=torch.index_select(value,1,new_data_cell_index)
        elif key=='boundary':
          new_dataset[key]=value
        else:
          new_dataset[key] = torch.index_select(value,1,new_data_index)
      cell_renum_dict = {}
      new_cells = np.empty_like(dataset['cells'][0])
      for i in range(new_dataset['mesh_pos'].shape[1]):
        cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i
      for j in range(dataset['cells'].shape[1]):
        cell = dataset['cells'][0][j]
        for node_index in range(cell.shape[0]):
          new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
      new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 ) 
      new_dataset['cells'] = torch.index_select(torch.from_numpy(new_cells),1,new_data_cell_index)

      cells_node = new_dataset['cells'][0]
      mesh_pos = new_dataset['mesh_pos'][0]
      new_dataset['centroid'] = ((torch.index_select(mesh_pos,0,cells_node[:,0])+torch.index_select(mesh_pos,0,cells_node[:,1])+torch.index_select(mesh_pos,0,cells_node[:,2]))/3.).view(1,-1,2)
      for key,value in new_dataset.items():
        dataset[key] = value.numpy()
        new_dataset[key] = value.numpy()
      #new_dataset = reorder_boundaryu_to_front(dataset) 
      new_dataset['cells'] = new_dataset['cells'][0:1,:,:]
      new_dataset['mesh_pos'] = new_dataset['mesh_pos'][0:1,:,:]
      new_dataset['node_type'] = new_dataset['node_type'][0:1,:,:]
      write_tfrecord_one_with_writer(writer,new_dataset,mode='cylinder_flow')
      print('origin datasets No.{} has been parsed mesh\n'.format(index_num))    

if __name__ == '__main__':
    # choose wether to transform whole datasets into h5 file
    
    tf.enable_resource_variables()
    tf.enable_eager_execution()
    pickl_path = path['pickl_save_path']
    tf_datasetPath = path['tf_datasetPath']
    #tf_datasetPath='/home/litianyu/mycode/repos-py/MeshGraphnets/pytorch/meshgraphnets-main/datasets/airfoil'
    #tf_datasetPath='/root/share/meshgraphnets/datasets/airfoil'
    numofsd = 2
    os.makedirs(path['tf_datasetPath'], exist_ok=True)

    for split in ['train','valid','test']:
        ds = load_dataset(tf_datasetPath, split)

        if path['saving_h5']:
          save_path=path['h5_save_path']+'_'+ model['name']+'_'+split+'_'+'.h5'
          save_path1=path['h5_save_path']+'_'+ model['name']+'_'+split +'_'+'hor_split1'+'.h5'
          save_path2=path['h5_save_path']+'_'+ model['name']+'_'+split +'_'+'hor_split2'+'.h5'
          f = h5py.File(save_path, "w")
          print(save_path)
        elif path['h5_sep']:
          f1 = h5py.File(save_path1, "w")
          f2 = h5py.File(save_path2, "w")
          print(save_path1)
          print(save_path2)
        #d = tf.data.make_one_shot_iterator(ds).zget_next()
        rearrange_frame_sp_1 = []
        rearrange_frame_sp_2 = []
        #parse_reshape(ds)
        raw_data = {}
        tf_saving_mesh_path=path['mesh_save_path']+'_'+  model['name']+'_'+ split+'.tfrecord'
        with tf.io.TFRecordWriter(tf_saving_mesh_path) as writer:
          for index, d in enumerate(ds):
              if 'mesh_pos' in d:
                mesh_pos = d['mesh_pos'].numpy()
                raw_data['mesh_pos']=mesh_pos
              if 'node_type' in d:
                node_type = d['node_type'].numpy()
                raw_data['node_type']=node_type
              if 'velocity' in d:
                velocity = d['velocity'].numpy()
                raw_data['velocity']=velocity
              if 'cells' in d:
                cells = d['cells'].numpy()
                raw_data['cells']=cells
              if 'density' in d:
                density =  d['density'].numpy()
                raw_data['density']=density
              if 'pressure' in d:
                pressure = d['pressure'].numpy()
                raw_data['pressure']=pressure
              if (path['renum_origin_dataset']):
                parse_origin_dataset(raw_data,unorder=False,index_num=index,plot=None,writer=writer)
              # if index%889==0 and index>0:  
              if True and index>0: 
                dataset = raw_data
                # dataset,rtvalue_renum = renum_data(dataset,True,index,"cell")
                # dataset,rtvalue_renum = renum_data(dataset,False,index,None)
                # if not rtvalue_renum:
                #   raise ValueError('InvalidArgumentError')
                rearrange_frame_1 = {}
                rearrange_frame_2 = {}
                if(path['plot_order']):
                    fig = plt.figure(figsize=(4,3))  # 创建画布
                    mesh_pos = dataset['mesh_pos'][0]
                    display_pos_list=[mesh_pos[0],mesh_pos[1],mesh_pos[2]]
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                    ax2.cla()
                    ax2.set_aspect('equal')
                    #bb_min = mesh['velocity'].min(axis=(0, 1))
                    #bb_max = mesh['velocity'].max(axis=(0, 1))
                    mesh_pos = dataset['mesh_pos'][0]
                    faces = dataset['cells'][0]
                    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
                    #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
                    ax2.triplot(triang, 'ko-', ms=0.5, lw=0.3)
                    #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)

                    def animate(num):
                      if num < mesh_pos.shape[0]:
                        display_pos_list.append(mesh_pos[num])
                      if num%3 ==0 and num >0:
                          display_pos = np.array(display_pos_list)
                          p1 = ax1.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
                          ax1.legend(['node_pos'], loc=2, fontsize=10)
                    ani = animation.FuncAnimation(fig, animate, frames=dataset['mesh_pos'][0].shape[0], interval=100) 
                    plt.show(block=True)  
                    ani.save("order"+"train.gif", writer='pillow')  
                    
              if(path['stastic']):
                stastic_nodeface_type(dataset['node_type'][0])
              if(path['saving_origin']):
                rtval = extract_mesh_state(raw_data,writer,index,mode=path['mode'])
                if not rtval:
                  print("parse error")
                  exit()
              # if(path['mask_features']):
              #   import plot_tfrecord as pltf
              #   rt_dataset = parser.mask_features(dataset,'velocity',0.1)
              #   pltf.plot_tfrecord_tmp(rt_dataset)
                
              if(path['print_tf']):
                with tf.Session() as sess:
                  velocity_t = tf.convert_to_tensor(velocity[0])
                  node_type_t = node_type[0][:,0].tolist()
                  res = tf.one_hot(indices=node_type_t, depth=NodeType.SIZE)
                  node_features = tf.concat([velocity_t, res], axis=-1)
                  print (sess.run(ds))
                  print (sess.run(res))
                  print (sess.run(node_features))
                for i in range(node_type.shape[0]):
                  print(i)
                  stastic(node_type[i])

              if path['saving_sp_tf']:  
                start_time = time.time()  
                rt_traj = seprate_cells(mesh_pos,cells,node_type,density,pressure,velocity,index)
                
                end_time = time.time()
                
                time_span = end_time - start_time
                
                print('dataset`s frame index is{0}, done. Cost time:{1}:'.format(index,time_span))
                
                if(path['saving_pickl']):
                  pickle_save(pickl_path,rearrange_frame_1)
                  
                if(path['saving_sp_tf_single']):
                  tf_save_path1=path['tfrecord_sp']+ model['name']+'_'+split+'_sp1'+'.tfrecord'
                  tf_save_path2=path['tfrecord_sp']+ model['name']+'_'+split+'_sp2'+'.tfrecord'     
                  write_tfrecord_one(tf_save_path1,rt_traj[0])
                  write_tfrecord_one(tf_save_path2,rt_traj[1])
                  if(index == 2):
                    break
                rearrange_frame_sp_1.append(rt_traj[0])
                rearrange_frame_sp_2.append(rt_traj[1])
                
        print('datasets {} has been extracted mesh\n'.format(split))     
        if path['saving_sp_tf_mp']:
          rearrange_frame_sp = [rearrange_frame_sp_1,rearrange_frame_sp_2]
          tf_save_path1=path['tfrecord_sp']+ model['name']+'_'+split+'_sp1'+'.tfrecord'
          tf_save_path2=path['tfrecord_sp']+ model['name']+'_'+split+'_sp2'+'.tfrecord'
          
          write_tfrecord_mp(tf_save_path1,tf_save_path2,rearrange_frame_sp)

            #print('splited mesh_pos1 is: ',mesh_pos_splited_1.shape)
            #print('splited mesh_pos2 is: ',mesh_pos_splited_2.shape)
        '''
            data = ("mesh_pos", "node_type", "velocity", "cells", "pressure")

            data_1 = ("mesh_pos", "node_type", "velocity_splited_1", "cells", "pressure")

            data_2 = ("mesh_pos", "node_type", "velocity_splited_2", "cells", "pressure")
            
            # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
            if path['saving_h5']:
              print('**************Now setting whole datasets into h5 file {0}******************'.format(index))
              s = str(index)
              g = f.create_group(s)
              for k in data:
                g[k] = eval(k)
              #print(index)
            else:
                print('**************Now spliting whole datasets into %d h5 file******************')
                g_1 = f1.create_group(str(index))
                g_2 = f2.create_group(str(index))
                for k in data_1:
                  g_1[k] = eval(k)
                for k in data_2:
                  g_2[k] = eval(k)
                print(index)
        f1.close()
        f2.close()
        f.close()'''
