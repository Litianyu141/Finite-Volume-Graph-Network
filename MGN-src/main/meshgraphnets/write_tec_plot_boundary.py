#2 -*- encoding: utf-8 -*-
'''
@File    :   write_tec_plot_boundary.py
@Author  :   litianyu 
@Version :   1.0
@Contact :   lty1040808318@163.com
'''
import json
import os
import sys
sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
from ast import Pass
from turtle import circle
import torch
import numpy as np
import pickle
import enum
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
matplotlib.use('Agg')
import pandas as pd
import circle_fit as cf
from circle_fit import hyper_fit
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

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
    
def triangles_to_faces(faces, deform=False):
    """Computes mesh edges from triangles."""
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
        unique_edges = torch.stack((receivers,senders), dim=1)
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers,'unique_edges':unique_edges}
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
    
def formatnp(data):
    '''
    Generate appropriate format string for numpy array

    Argument:
        - data: a list of numpy array
    '''

    dataForm = []
    for i in range(len(data)):
        if np.issubsctype(data[i], np.integer):
            dataForm.append(' {:d}'.format(data[i]))
        else:
            dataForm.append(' {:e}'.format(data[i]))
        if ((i+1)%3==0):
            dataForm.append('\n')
        if (i == (len(data)-1) and (i+1)%3>0):
            dataForm.append('\n')
    return ' '.join(dataForm)
def formatnp_c(data):
    '''
    Generate appropriate format string for numpy array

    Argument:
        - data: a list of numpy array
    '''

    dataForm = []
    for i in range(len(data)):
        if np.issubsctype(data[i], np.integer):
            dataForm.append(' {:d}'.format(data[i]))
        else:
            dataForm.append(' {:e}'.format(data[i]))
        if ((i+1)%3==0):
            dataForm.append('\n')
    return ' '.join(dataForm)
def formatnp_f(data):
    '''
    Generate appropriate format string for numpy array

    Argument:
        - data: a list of numpy array
    '''

    dataForm = []
    for i in range(len(data)):
        if np.issubsctype(data[i], np.integer):
            dataForm.append(' {:d}'.format(data[i]))
        else:
            dataForm.append(' {:e}'.format(data[i]))
        if ((i+1)%2==0):
            dataForm.append('\n')
    return ' '.join(dataForm)

def write_cell_index(Cells,writer):
    for index in range(Cells.shape[0]):
        writer.write(formatnp_c(Cells[index]))
        
def write_face_index(faces,writer):
    for index in range(faces.shape[0]):
        writer.write(formatnp_f(faces[index]))
        
def write_tecplotzone(filename='flowcfdgcn.dat',datasets=None,time_step_length=100,has_cell_centered=False):
    time_avg_start=100
    time_sum_velocity=np.zeros_like(datasets[0]['velocity'])[1,:,:]
    time_sum_pressure=np.zeros_like(datasets[0]['pressure'])[1,:,:]
    time_avg_velocity=np.zeros_like(datasets[0]['velocity'])
    time_avg_pressure=np.zeros_like(datasets[0]['pressure'])
    
    target_time_sum_velocity=np.zeros_like(datasets[0]['velocity'])[1,:,:]
    target_time_sum_pressure=np.zeros_like(datasets[0]['pressure'])[1,:,:]
    target_time_avg_velocity=np.zeros_like(datasets[0]['velocity'])
    target_time_avg_pressure=np.zeros_like(datasets[0]['pressure'])
    
    for i in range(time_avg_start,datasets[0]['velocity'].shape[0]):
        time_sum_velocity+=datasets[0]['velocity'][i]
        time_avg_velocity[i]=(time_sum_velocity/(i-time_avg_start+1))
        time_sum_pressure+=datasets[0]['pressure'][i]
        time_avg_pressure[i]=(time_sum_pressure/(i-time_avg_start+1))
        
        target_time_sum_velocity+=datasets[0]['target|UVP'][i,:,0:2]
        target_time_avg_velocity[i]=(target_time_sum_velocity/(i-time_avg_start+1))
        target_time_sum_pressure+=datasets[0]['target|UVP'][i,:,2:3]
        target_time_avg_pressure[i]=(target_time_sum_pressure/(i-time_avg_start+1))
        
    with open(filename, 'w') as f:

        f.write('TITLE = "Visualization of the volumetric solution"\n')
        f.write('VARIABLES = "X"\n"Y"\n"U"\n"V"\n"P"\n"avgU"\n"avgV"\n"avgP"\n"target|U"\n"target|V"\n"target|P"\n"target|avgU"\n"target|avgV"\n"target|avgP"\n')
        for i in range(time_step_length):
            for zone in datasets:
                zonename = zone['zonename']
                if zonename =='Fluid':
                    f.write('ZONE T="{0}"\n'.format(zonename))
                    X = zone['mesh_pos'][i,:,0]
                    Y = zone['mesh_pos'][i,:,1]
                    U = zone['velocity'][i,:,0]
                    V = zone['velocity'][i,:,1]
                    P = zone['pressure'][i,:,0]
                    avgU = time_avg_velocity[i,:,0]
                    avgV = time_avg_velocity[i,:,1]
                    avgP = time_avg_pressure[i,:,0]
                    target_U = zone['target|UVP'][i,:,0]
                    target_V = zone['target|UVP'][i,:,1]
                    target_P = zone['target|UVP'][i,:,2]
                    avgtarget_U = target_time_avg_velocity[i,:,0]
                    avgtarget_V = target_time_avg_velocity[i,:,1]
                    avgtarget_P = target_time_avg_pressure[i,:,0]
                    field = np.concatenate((X,Y,U,V,P,avgU,avgV,avgP,target_U,target_V,target_P,avgtarget_U,avgtarget_V,avgtarget_P),axis=0)
                    Cells = zone['cells'][i,:,:]+1
                
                    f.write(' STRANDID=1, SOLUTIONTIME={0}\n'.format(0.01*i))
                    f.write(f' Nodes={X.size}, Elements={Cells.shape[0]}, ''ZONETYPE=FETRIANGLE\n')
                    f.write(' DATAPACKING=BLOCK\n') 
                    if has_cell_centered:
                        f.write(' VARLOCATION=([3,4,5,6,7,8,9,10,11,12,13,14]=CELLCENTERED)\n') 
                    else:
                        f.write(' VARLOCATION=([3,4,5,6,7,8]=NODAL)\n') 
                    f.write(' DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n') 
                    f.write(formatnp(field))
                    f.write(" ")
                    write_cell_index(Cells,f)
        print('saved tecplot file at '+filename)
        '''
        for node, field in zip(x, fields):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'f'{field[0].item()}\t{field[1].item()}\t'f'{field[2].item()}\n')

        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            #if len(elem) == 3:
                # repeat last vertex if triangle
            #    f.write(f'\t{elem[-1]+1}')
            f.write('\n')
        '''         
        
def write_tecplot_ascii_nodal(raw_data,is_tfrecord,pkl_path,saving_path):
    cylinder_pos = []
    cylinder_velocity= []
    cylinder_pressure = []
    cylinder_index = []
    cylinder = {}
    if is_tfrecord:
      dataset= raw_data
    else:
      with open(pkl_path, 'rb') as fp:
        dataset = pickle.load(fp)
    for j in range(600):
      new_pos_dict = {}
      mesh_pos = dataset['mesh_pos'][j]
      coor_y = dataset['mesh_pos'][j,:,1]
      mask_F = np.full(coor_y.shape,False)
      mask_T = np.full(coor_y.shape,True)
      node_type = dataset['node_type'][j,:,0]
      mask_of_coor = np.where((node_type==NodeType.WALL_BOUNDARY)&(coor_y>np.min(coor_y))&(coor_y<np.max(coor_y)),mask_T,mask_F)
      mask_of_coor_index = np.argwhere((node_type==NodeType.WALL_BOUNDARY)&(coor_y>np.min(coor_y))&(coor_y<np.max(coor_y)))
      cylinder_x = dataset['mesh_pos'][j,:,0][mask_of_coor]
      cylinder_u = dataset['velocity'][j,:,0][mask_of_coor]
      cylinder_y = coor_y[mask_of_coor]
      cylinder_v = dataset['velocity'][j,:,1][mask_of_coor]
      cylinder_p = dataset['pressure'][j,:,0][mask_of_coor]
      coor = np.stack((cylinder_x,cylinder_y),axis=-1)
      cylinder_speed = np.stack((cylinder_u,cylinder_v),axis=-1)
      
      
      cylinder_pos.append(coor)
      cylinder_velocity.append(cylinder_speed)
      cylinder_pressure.append(cylinder_p)
      
      for index in range(coor.shape[0]):
        new_pos_dict[str(coor[index])] = index
      cells_node = torch.from_numpy(dataset['cells'][j]).to(torch.int32)
      decomposed_cells = triangles_to_faces(cells_node)
      senders = decomposed_cells['senders']
      receivers = decomposed_cells['receivers']
      mask_F = np.full(senders.shape,False)
      mask_T = np.full(senders.shape,True)
      mask_index_s = np.isin(senders,mask_of_coor_index)
      mask_index_r = np.isin(receivers,mask_of_coor_index)
      
      mask_index_of_face = np.where((mask_index_s)&(mask_index_r),mask_T,mask_F)
      
      senders = senders[mask_index_of_face]
      receivers = receivers[mask_index_of_face]
      senders_f = []
      receivers_f = []
      for i in range(senders.shape[0]):
        senders_f.append(new_pos_dict[str(mesh_pos[senders[i]])])
        receivers_f.append(new_pos_dict[str(mesh_pos[receivers[i]])])
      cylinder_boundary_face = np.stack((np.asarray(senders_f),np.asarray(receivers_f)),axis=-1)
      cylinder_index.append(cylinder_boundary_face)
    dataset['zonename']='Fluid'
    flow_zone = dataset
    cylinder['zonename'] = 'Cylinder_Boundary'
    cylinder['mesh_pos'] = np.asarray(cylinder_pos)
    cylinder['velocity'] = np.asarray(cylinder_velocity)
    cylinder['pressure'] = np.expand_dims(np.asarray(cylinder_pressure),-1)
    cylinder['face'] = np.asarray(cylinder_index)
    cylinder_zone = cylinder
    tec_saving_path = saving_path
    write_tecplotzone(tec_saving_path,[flow_zone,cylinder_zone]) 
      
def rearrange_dict(zone):
    '''transform dict to list, so pandas dataframe can handle it properly'''
    dict_list = []
    build = False
    for k, v in zone.items():
        if k =='zonename' or k =='mean_u' or k =='relonyds_num' or k =='cylinder_D':
            continue
        if v.shape[2]>1:
            for j in range(v.shape[2]):
                for index in range(zone['mesh_pos'].shape[0]):
                    dict_new = {}
                    if k =='zonename' or k =='mean_u' or k =='relonyds_num' or k =='cylinder_D':
                        continue
                    elif k == 'centroid' or k == 'cell_area':
                        dict_new[k+str(j)] = v[0][:,j]
                    else:
                        dict_new[k+str(j)] = v[index][:,j]
                    if not build:
                        dict_list.append(dict_new)  
                build=True
                for index in range(zone['mesh_pos'].shape[0]):
                    if k =='zonename' or k =='mean_u' or k =='relonyds_num' or k =='cylinder_D':
                        continue
                    elif k == 'centroid' or k == 'cell_area':
                        dict_list[index][k+str(j)]=v[0][:,j]
                    else:
                        dict_list[index][k+str(j)]=v[index][:,j]
        else:
            for index in range(zone['mesh_pos'].shape[0]):
                if k =='zonename' or k =='mean_u' or k =='relonyds_num' or k =='cylinder_D':
                    continue
                elif k == 'centroid' or k == 'cell_area':
                    dict_list[index][k]=v[0][:,0]
                else:
                    dict_list[index][k]=v[index][:,0]
        build=True
    return dict_list

def extract_theta(cylinder_zone):
    
    mesh_pos = cylinder_zone['mesh_pos'][0]
    xc,yc,R,_=hyper_fit(np.asarray(mesh_pos))
    #cf.plot_data_circle(mesh_pos[:,0],mesh_pos[:,1],xc,yc,r)
    # circle_center = torch.from_numpy(np.array([xc,yc],dtype=np.float32)).view(1,2).repeat(cylinder_zone['mesh_pos'].shape[1],1)
    face_center_pos = (torch.index_select(torch.from_numpy(mesh_pos),0,torch.from_numpy(cylinder_zone['face'][0][:,0]))+torch.index_select(torch.from_numpy(mesh_pos),0,torch.from_numpy(cylinder_zone['face'][0][:,1])))/2.
    
    _,wingleft_index = torch.max(face_center_pos[:,0:1],dim=0)
    _,winglright_index = torch.min(face_center_pos[:,0:1],dim=0)
    
    circle_center_pos = (face_center_pos[wingleft_index]+face_center_pos[winglright_index])/2.
    circle_center = circle_center_pos.view(1,2).repeat(cylinder_zone['mesh_pos'].shape[1],1)
    # xcf,ycf,rf,_=hyper_fit(face_center_pos.numpy())
    #cf.plot_data_circle(face_center_pos[:,0],face_center_pos[:,1],xcf,ycf,rf)
    _,pivot_index = torch.max(face_center_pos[:,0:1],dim=0)
    c_2_edge = face_center_pos-circle_center
    # _,pivot_index = torch.min(c_2_edge[:,0:1],dim=0)
    pivot = c_2_edge[pivot_index]
    cos_theta = torch.sum(c_2_edge*pivot.repeat(c_2_edge.shape[0],1),dim=1)/(torch.norm(c_2_edge,p=2,dim=1)*torch.norm(pivot.repeat(c_2_edge.shape[0],1),p=2,dim=1))
    
    # 计算 theta 的新方法
    theta = torch.atan2(c_2_edge[:, 1], c_2_edge[:, 0])

    # 将负角度值转换为 [0, 2π] 范围内的正值
    theta[theta < 0] += 2 * np.pi
    
    cylinder_D = torch.norm(face_center_pos[wingleft_index]-face_center_pos[winglright_index])
    cylinder_zone['cylinder_D'] = cylinder_D.numpy()
    cylinder_zone['cos_theta'] = cos_theta.view(1,-1,1).repeat(cylinder_zone['mesh_pos'].shape[0],1,1)
    cylinder_zone['theta'] = theta.view(1,-1,1).repeat(cylinder_zone['mesh_pos'].shape[0],1,1)
    
    return cylinder_zone

def plot_boundary_pressure(boundary_zone,saving_path=None,plot_index=None):
    
    boundary_zone_to_plot = {}
    boundary_zone_to_plot['mean_u'] = boundary_zone['mean_u']
    boundary_zone_to_plot['relonyds_num'] = boundary_zone['relonyds_num']
    boundary_zone_to_plot['mesh_pos'] = boundary_zone['mesh_pos']
    boundary_zone_to_plot['pressure'] = boundary_zone['pressure']
    boundary_zone_to_plot['target|pressure'] = boundary_zone['target|pressure']
    boundary_zone_to_plot['face'] = boundary_zone['face']

    boundary_zone_to_plot = extract_theta(boundary_zone_to_plot)
    frame_list=[]
    boundary_zone_list = rearrange_dict(boundary_zone_to_plot)
    
    path = os.path.split(saving_path)[0]
    Re = boundary_zone['relonyds_num']
    
    for index in range(len(boundary_zone_list)):
        frame_list.append(pd.DataFrame(boundary_zone_list[index]).sort_values(by='theta',ascending=True))

    
    if plot_index is not None:
        if type(plot_index) is int:
            from matplotlib import rcParams
            from matplotlib.ticker import AutoMinorLocator, MaxNLocator
            config = {
                "font.size": 20,
                "mathtext.fontset":'stix',
                "font.serif": ['SimSun'],
            }
            rcParams.update(config)
            plt.figure(figsize=(8,4))
            ax = plt.subplot(111)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # 设置Y轴的主要刻度定位器为6个刻度
            ax.yaxis.set_major_locator(MaxNLocator(5))
            
            # 添加X轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            
            # 添加Y轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            
            ax.tick_params(which='both', direction='in')
            
            ax.set_xlim(left=0)
            
            # 计算从第100步到最后一步的平均压力值
            start_step = 100  # 开始步骤
            end_step = len(frame_list)  # 结束步骤，即最后一步
            avg_pressure = np.mean([frame_list[i]["pressure"].values for i in range(start_step, end_step)], axis=0)
            avg_target_pressure = np.mean([frame_list[i]["target|pressure"].values for i in range(start_step, end_step)], axis=0)

            # 绘制平均压力分布曲线
            plt.plot(
                range(len(avg_target_pressure) + 1),
                list(avg_target_pressure) + [avg_target_pressure[0]],
                'k-', linewidth=2, label="CFD Average"  # 使用黑色加粗线表示平均CFD值
            )
            plt.plot(
                range(len(avg_pressure) + 1),
                list(avg_pressure) + [avg_pressure[0]],
                'r--', marker='o', markersize=4, markevery=1, linewidth=2, label="Prediction Average"  # 使用虚线和更大的叉叉标记表示平均预测值
            )

            plt.legend(fontsize=12, loc='upper right')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            # 设置 x 轴和 y 轴的单位
            plt.xlabel("$θ$ (° degrees)", fontsize=17)
            plt.ylabel("Pressure", fontsize=17)

            # 计算每个刻度之间的步长
            number_of_ticks = len(frame_list[plot_index]["pressure"].values)
            step_size = 360 / (number_of_ticks)
                                
            ticks_range=np.arange(0, 361, step=round(step_size * (len(frame_list[plot_index]["pressure"].values) / 5)))
            theta_values_pos = np.linspace(0, number_of_ticks, num=int(len(ticks_range)), endpoint=True)
            for j in range(len(theta_values_pos)):
                theta_values_pos[j]=int(theta_values_pos[j])
            # 设置 x 轴的刻度标签以显示 0-360° 的角度，包括最右端的 360°
            plt.xticks(theta_values_pos, ticks_range)
            plt.subplots_adjust(bottom=0.2)  # 增加底部边距
            plt.savefig(path+f'/Re(%2e)_'%Re+'Pressure distubation.png',dpi=300)
            plt.close()
            # plt.show()
  
def write_tecplot_ascii_cell_centered(raw_data,saving_path,plot_boundary_p_time_step=None,save_tec=False,plot_boundary=False):
    
    # boundary zone
    if save_tec or plot_boundary_p_time_step is not None:
        cylinder_zone =  extract_boundary_thread(raw_data=raw_data,
                                                save_path=saving_path,
                                                plot_boundary=plot_boundary)
    
    # plot boundary zone
    if plot_boundary_p_time_step is not None:
        plot_boundary_pressure(cylinder_zone,
                               saving_path,
                               plot_boundary_p_time_step)
    
    # write interior zone and boundary zone to tecplot file
    raw_data['zonename']='Fluid'
    flow_zone = raw_data
    tec_saving_path = saving_path
    if save_tec:
        write_tecplotzone(filename=tec_saving_path,
                            datasets=[flow_zone,cylinder_zone],
                            time_step_length=raw_data['velocity'].shape[0],
                            has_cell_centered=True)    
            
def extract_boundary_thread(raw_data,save_path=None,plot_boundary=False):
    # surface zone
    cylinder = {}
    cylinder_node_pos= []
    target_cylinder_face_velocity_list = []
    target_cylinder_face_pressure_list = []
    cylinder_face_velocity_list = []
    cylinder_face_pressure_list = []
    cylinder_cell_centroid_list = []
    cylinder_cell_area_list = []
    cylinder_index = []
    new_pos_dict = {}
    cells = torch.from_numpy(raw_data['cells'][0]).to(torch.int32)
    node_type = torch.from_numpy(raw_data['node_type'][0,:,:]).to(torch.int32).squeeze(1)
    mesh_pos = torch.from_numpy(raw_data['mesh_pos'][0])
    coor_x = raw_data['mesh_pos'][0,:,0]
    coor_y = raw_data['mesh_pos'][0,:,1]
    mask_F = np.full(coor_y.shape,False)
    mask_T = np.full(coor_y.shape,True)

    # maksing boundary points
    mask_of_coor = np.where((node_type==NodeType.WALL_BOUNDARY)&(coor_y>np.min(coor_y))&(coor_y<np.max(coor_y))&(coor_x<np.max(coor_x))&(coor_x>np.min(coor_x)),mask_T,mask_F)
    mask_of_coor_index = np.argwhere((node_type==NodeType.WALL_BOUNDARY)&(coor_y>np.min(coor_y))&(coor_y<np.max(coor_y))&(coor_x<np.max(coor_x))&(coor_x>np.min(coor_x)))
    cylinder_x = raw_data['mesh_pos'][0,:,0][mask_of_coor]
    cylinder_y = coor_y[mask_of_coor]
    coor = raw_data['boundary_mesh_pos'][0]
    cylinder_node_pos.append(coor)
    
    '''for checking boundary'''
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.scatter(coor[:,0],coor[:,1],c='red',linewidths=1)
    # plt.show()
    plt.close()
    '''for checking boundary'''
    
    for index in range(coor.shape[0]):
        new_pos_dict[str(coor[index])] = index
        
    # cylinder face velocity
    cell_3node_pos =[torch.index_select(mesh_pos,0,cells[:,0]),torch.index_select(mesh_pos,0,cells[:,1]),torch.index_select(mesh_pos,0,cells[:,2])]
    cell_type = torch.from_numpy(raw_data['cell_type'][0,:,0])
    zone_top_coor_y = torch.max(mesh_pos[:,1])
    zone_bottom_coor_y = torch.min(mesh_pos[:,1])
    
    cylinder_face_velocity = raw_data['boundary_velocity']
    
    target_cylinder_face_velocity_list = raw_data['target_cylinder|velocity']
    
    cylinder_face_velocity_list=cylinder_face_velocity
    
    cylinder_face_pressure = raw_data['boundary_pressure']
    cylinder_face_pressure_list = cylinder_face_pressure
    
    target_cylinder_face_pressure_list = raw_data['target_cylinder|pressure']
    
    cylinder_cell_centriod = (torch.from_numpy(raw_data['centroid'][:,:]).view(1,-1,2)[:,(cell_type==NodeType.WALL_BOUNDARY)&(cell_3node_pos[0][:,1]>zone_bottom_coor_y)&(cell_3node_pos[0][:,1]<zone_top_coor_y)&(cell_3node_pos[1][:,1]>zone_bottom_coor_y)&(cell_3node_pos[1][:,1]<zone_top_coor_y)&(cell_3node_pos[2][:,1]>zone_bottom_coor_y)&(cell_3node_pos[2][:,1]<zone_top_coor_y),:]).numpy()
    cylinder_cell_centroid_list=cylinder_cell_centriod

    cylinder_cell_area = (torch.from_numpy(raw_data['cell_area'][:,:]).view(1,-1,1)[:,(cell_type==NodeType.WALL_BOUNDARY)&(cell_3node_pos[0][:,1]>zone_bottom_coor_y)&(cell_3node_pos[0][:,1]<zone_top_coor_y)&(cell_3node_pos[1][:,1]>zone_bottom_coor_y)&(cell_3node_pos[1][:,1]<zone_top_coor_y)&(cell_3node_pos[2][:,1]>zone_bottom_coor_y)&(cell_3node_pos[2][:,1]<zone_top_coor_y),:]).numpy()
    cylinder_cell_area_list=cylinder_cell_area
    
    # reorder the node index which store in array boundary_face
    senders = raw_data['boundary_face'][0][:,0]
    receivers = raw_data['boundary_face'][0][:,1]
    senders_f = []
    receivers_f = []
    for i in range(senders.shape[0]):
        senders_f.append(new_pos_dict[str(mesh_pos[senders[i]].numpy())])
        receivers_f.append(new_pos_dict[str(mesh_pos[receivers[i]].numpy())])
    cylinder_boundary_face = np.stack((np.asarray(senders_f),np.asarray(receivers_f)),axis=-1)
    cylinder_index.append(cylinder_boundary_face)

    # obj surface zone
    cylinder['zonename'] = 'Cylinder_Boundary'
    cylinder['mean_u'] = raw_data['mean_u']
    cylinder['relonyds_num'] = raw_data['relonyds_num']
    cylinder['mesh_pos'] = np.asarray(cylinder_node_pos).repeat(raw_data['velocity'].shape[0],axis=0)
    cylinder['velocity'] = np.asarray(cylinder_face_velocity_list)
    cylinder['target|velocity'] = np.asarray(target_cylinder_face_velocity_list)
    cylinder['pressure'] = np.asarray(cylinder_face_pressure_list)
    cylinder['target|pressure'] = np.asarray(target_cylinder_face_pressure_list)
    cylinder['centroid'] = np.asarray(cylinder_cell_centroid_list).repeat(raw_data['velocity'].shape[0],axis=0)
    cylinder['cell_area'] = np.asarray(cylinder_cell_area_list).repeat(raw_data['velocity'].shape[0],axis=0)
    cylinder['face'] = np.asarray(cylinder_index).repeat(raw_data['velocity'].shape[0],axis=0)
    cylinder['edge_index'] = torch.from_numpy(np.asarray(cylinder_index).repeat(raw_data['velocity'].shape[0],axis=0)).transpose(1,2).numpy()
    cylinder['target_edge_UVP'] = raw_data['target_edge_UVP']
    cylinder['cylinder_boundary_cells'] = raw_data['cylinder_boundary_cells']
    cylinder['cylinder_boundary_cell_unv'] = raw_data['cylinder_boundary_cell_unv'] 
    cylinder['cylinder_boundary_cell_area'] = raw_data['cylinder_boundary_cell_area']
    cylinder['cylinder_boundary_cell_face'] = raw_data['cylinder_boundary_cell_face'] 
    cylinder['predicted_edge_UVP'] = raw_data['predicted_edge_UVP'] 
    
    cylinder['target_cylinder_uvp'] = np.concatenate((raw_data['target_cylinder|velocity'],raw_data['target_cylinder|pressure']),axis=2)
    
    cylinder['face_length'] = raw_data['face_length']
    cylinder_zone = cylinder
    
    '''plot CL and CD at boundary'''
    if plot_boundary:
        cylinder_zone = vortex_on_boundary_cell(cylinder_zone,plotting_pressure_force=True, save_path=save_path)
        
    return cylinder_zone

def calc_edge_unv(edge_index,mesh_pos,centroid):
    
        senders = torch.from_numpy(edge_index[:,0])
        receivers = torch.from_numpy(edge_index[:,1])
        #calculate unit norm vector
        # unv = torch.ones((edge_index.shape[0],2),dtype=torch.float32)
        pos_diff = torch.index_select(torch.from_numpy(mesh_pos),0,senders)-torch.index_select(torch.from_numpy(mesh_pos),0,receivers)
        unv = torch.stack((-pos_diff[:,1],pos_diff[:,0]),dim=-1)
        face_length = torch.norm(pos_diff,dim=1)

        # unv[:,1] = -(pos_diff[:,0]/pos_diff[:,1])
        deinf = torch.tensor([0,1],dtype=torch.float32).repeat(unv.shape[0],1)
        unv = torch.where((torch.isinf(unv)),deinf,unv)
        unv = unv/torch.norm(unv,dim=1).view(-1,1)
        face_center_pos = (torch.index_select(torch.from_numpy(mesh_pos),0,senders)+torch.index_select(torch.from_numpy(mesh_pos),0,receivers))/2.
        
        _,wingleft_index = torch.max(torch.from_numpy(mesh_pos)[:,0:1],dim=0)
        _,winglright_index = torch.min(torch.from_numpy(mesh_pos)[:,0:1],dim=0)
        
        cylinder_center = ((torch.from_numpy(mesh_pos)[wingleft_index]+torch.from_numpy(mesh_pos)[winglright_index])/2.).view(-1,2)
        
        c_f = cylinder_center-face_center_pos
        edge_uv_times_ccv = c_f[:,0]*unv[:,0]+c_f[:,1]*unv[:,1]
        mask = torch.logical_or((edge_uv_times_ccv)>0,torch.full(edge_uv_times_ccv.shape,False))
        unv = torch.where(torch.stack((mask,mask),dim=-1),unv,unv*(-1.))
        
        return unv,face_length.view(-1,1)

def vortex_on_boundary_cell(boundary_zone,plotting_pressure_force=True, save_path=None):
    vortex_on_boundary = []
    target_vortex_on_boundary = []
    unv_on_boundary = []
    face_length_on_boundary = []
    F_t_list = []
    target_F_t_list=[]
    F_p_list = []
    target_F_p_list = []
    for rollout_step in range(boundary_zone['face'].shape[0]):
        cylinder_edge_unv,cylinder_edge_length = calc_edge_unv(edge_index=boundary_zone['face'][rollout_step],mesh_pos=boundary_zone['mesh_pos'][rollout_step],centroid=boundary_zone['centroid'][rollout_step])
        
        cylinder_boundary_cell_face = torch.from_numpy(boundary_zone['cylinder_boundary_cell_face'])[rollout_step]
        predicted_edge_UVP = torch.from_numpy(boundary_zone['predicted_edge_UVP'])[rollout_step]
        target_cylinder_edge_uvp = torch.from_numpy(boundary_zone['target_cylinder_uvp'][rollout_step])
        
        target_edge_uvp = torch.from_numpy(boundary_zone['target_edge_UVP'][rollout_step])
        #calculate unit norm vector
        cylinder_boundary_cell_unv = torch.from_numpy(boundary_zone['cylinder_boundary_cell_unv'][rollout_step])
        face_length = torch.from_numpy(boundary_zone['face_length'][rollout_step])
        cylinder_face_length = torch.stack((torch.index_select(face_length,0,cylinder_boundary_cell_face[0]),torch.index_select(face_length,0,cylinder_boundary_cell_face[1]),torch.index_select(face_length,0,cylinder_boundary_cell_face[2])),dim=0).transpose(0,1).squeeze(2)
        
        face_length_on_boundary.append(cylinder_edge_length.view(-1,1).numpy())

        unv_on_boundary.append(cylinder_boundary_cell_unv.numpy())
        
        cell_area = torch.from_numpy(boundary_zone['cylinder_boundary_cell_area'][rollout_step])
        velocity = torch.stack((torch.index_select(predicted_edge_UVP[:,0:2],0,cylinder_boundary_cell_face[0]),torch.index_select(predicted_edge_UVP[:,0:2],0,cylinder_boundary_cell_face[1]),torch.index_select(predicted_edge_UVP[:,0:2],0,cylinder_boundary_cell_face[2])),dim=0).transpose(0,1)
        
        target_velocity = torch.stack((torch.index_select(target_edge_uvp[:,0:2],0,cylinder_boundary_cell_face[0]),torch.index_select(target_edge_uvp[:,0:2],0,cylinder_boundary_cell_face[1]),torch.index_select(target_edge_uvp[:,0:2],0,cylinder_boundary_cell_face[2])),dim=0).transpose(0,1)
        
        Voterx_z = (1./cell_area[:,0])*(torch.sum(((cylinder_boundary_cell_unv[:,:,0]*velocity[:,:,1]-cylinder_boundary_cell_unv[:,:,1]*velocity[:,:,0])*cylinder_face_length),dim=1))/2.
        
        target_Voterx_z = (1./cell_area[:,0])*(torch.sum(((cylinder_boundary_cell_unv[:,:,0]*target_velocity[:,:,1]-cylinder_boundary_cell_unv[:,:,1]*target_velocity[:,:,0])*cylinder_face_length),dim=1))/2.
        
        vortex_on_boundary.append(Voterx_z.view(-1,1).numpy())
        target_vortex_on_boundary.append(target_Voterx_z.view(-1,1).numpy())
        
        F_t =torch.cat(((Voterx_z*cylinder_edge_unv[:,1]).view(-1,1),(Voterx_z*cylinder_edge_unv[:,0]).view(-1,1)),dim=1)*cylinder_edge_length
        F_t_list.append(F_t.numpy())
        
        target_F_t =torch.cat(((target_Voterx_z*cylinder_edge_unv[:,1]).view(-1,1),(target_Voterx_z*cylinder_edge_unv[:,0]).view(-1,1)),dim=1)*cylinder_edge_length
        target_F_t_list.append(target_F_t.numpy())
        
        F_p = torch.cat((torch.from_numpy(boundary_zone['pressure'][rollout_step])*cylinder_edge_unv[:,0:1],torch.from_numpy(boundary_zone['pressure'][rollout_step])*cylinder_edge_unv[:,1:2]),dim=1)*cylinder_edge_length
        F_p_list.append(F_p.numpy())
        
        target_F_p = torch.cat(((target_cylinder_edge_uvp[:,2:]*cylinder_edge_unv[:,0:1]),(target_cylinder_edge_uvp[:,2:]*cylinder_edge_unv[:,1:2])),dim=1)*cylinder_edge_length
        target_F_p_list.append(target_F_p.numpy())
        
        '''for checking boundary'''
        # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        # plt.scatter(boundary_zone['mesh_pos'][rollout_step,:,0],boundary_zone['mesh_pos'][rollout_step,:,1],c='red',linewidths=1)
        # face_center = (boundary_zone['mesh_pos'][0][boundary_zone['face'][rollout_step][:,0]]+boundary_zone['mesh_pos'][0][boundary_zone['face'][rollout_step][:,1]])/2.
        # plt.scatter(face_center[:,0],face_center[:,1],c='green',linewidths=1)
        
        # plt.quiver(face_center[:,0], face_center[:,1], cylinder_edge_unv[:,0], cylinder_edge_unv[:,1], angles='xy', scale_units='xy', scale=200, linewidth=0.5)
        
        # plt.scatter(boundary_zone['centroid'][rollout_step][:,0],boundary_zone['centroid'][rollout_step][:,1],c='blue',linewidths=1)
        # plt.show()
        '''for checking boundary'''
            
    boundary_zone = extract_theta(boundary_zone)    
    boundary_zone['voterx_z'] = np.asarray(vortex_on_boundary)
    boundary_zone['unv'] = np.asarray(unv_on_boundary)
    boundary_zone['face_length_on_boundary'] = np.asarray(face_length_on_boundary)
    boundary_zone['F_t'] = np.asarray(F_t_list)
    boundary_zone['F_p'] = np.asarray(F_p_list)
    boundary_zone['target_F_t'] = np.asarray(target_F_t_list)
    boundary_zone['target_F_p'] = np.asarray(target_F_p_list)
    
    
    if plotting_pressure_force:
        plot_F_pt(boundary_zone,mu=0.001,save_path=save_path)
    return boundary_zone

def polygon_area(coordinates):
    n = len(coordinates)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)

    return abs(area) / 2

def plot_F_pt(boundary_zone, mu=0.001, save_path=None):
    path = os.path.split(save_path)[0]

    Re = boundary_zone['relonyds_num']
    mu = torch.as_tensor(mu)

    f_px = torch.from_numpy(boundary_zone['F_p'])[:,:,0]
    f_py = torch.from_numpy(boundary_zone['F_p'])[:,:,1]
    f_tx = torch.from_numpy(boundary_zone['F_t'])[:,:,0]
    f_ty = torch.from_numpy(boundary_zone['F_t'])[:,:,1]

    target_f_px = torch.from_numpy(boundary_zone['target_F_p'])[:,:,0]
    target_f_py = torch.from_numpy(boundary_zone['target_F_p'])[:,:,1]
    target_f_tx = torch.from_numpy(boundary_zone['target_F_t'])[:,:,0]
    target_f_ty = torch.from_numpy(boundary_zone['target_F_t'])[:,:,1]

    F_px = torch.sum(f_px, dim=1)
    F_py = torch.sum(f_py, dim=1)
    taget_F_px = torch.sum(target_f_px, dim=1)
    taget_F_py =torch.sum(target_f_py, dim=1)

    F_tx = torch.sum(mu * f_tx, dim=1)
    F_ty = torch.sum(mu * f_ty, dim=1)
    taget_F_tx = torch.sum(mu * target_f_tx, dim=1)
    taget_F_ty = torch.sum(mu * target_f_ty, dim=1)

    from matplotlib import rcParams
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter
    config = {
        "font.size": 20,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    
    rcParams.update(config)
    plt.figure(figsize=(16,7))

    ax_pgf=plt.subplot(211)
    ax_pgf.spines['right'].set_visible(False)
    ax_pgf.spines['top'].set_visible(False)
    ax_pgf.set_xticklabels([])
    ax_pgf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # 设置Y轴的主要刻度定位器为4个刻度
    ax_pgf.yaxis.set_major_locator(MaxNLocator(4))
    
    # 添加X轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
    ax_pgf.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    # 添加Y轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
    ax_pgf.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax_pgf.tick_params(which='both', direction='in')
    
    ax_pgf.set_xlim(0, taget_F_px.size(0))
    
    plt.plot(taget_F_px, 'darkblue', linewidth=3, label="CFD x-dir")  # CFD结果 x-dir
    plt.plot(taget_F_py, 'k-', linewidth=3, label="CFD y-dir")  # CFD结果 y-dir
    plt.plot(F_px, 'g--', marker='o', markersize=5, markevery=10, linewidth=2, label="Prediction x-dir")  # 预测结果 x-dir
    plt.plot(F_py, 'r--', marker='o', markersize=5, markevery=10, linewidth=2, label="Prediction y-dir")  # 预测结果 y-dir
    
    # plt.legend(fontsize=10, loc='upper right')
    plt.yticks(fontsize=20)
    plt.ylabel("$D_p$", fontsize=20, fontstyle='italic', labelpad=40)

    
    ax_vf = plt.subplot(212)
    ax_vf.spines['right'].set_visible(False)
    ax_vf.spines['top'].set_visible(False)

    ax_vf.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    # 设置Y轴的主要刻度定位器为4个刻度
    ax_vf.yaxis.set_major_locator(MaxNLocator(4))
    
    # 添加X轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
    ax_vf.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    # 添加Y轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
    ax_vf.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax_vf.tick_params(which='both', direction='in')
    
    ax_vf.set_xlim(0, taget_F_tx.size(0))
    
    plt.plot(taget_F_tx, 'darkblue', linewidth=3, label="CFD x-dir")  # CFD结果 x-dir
    plt.plot(taget_F_ty, 'k-', linewidth=3, label="CFD y-dir")  # CFD结果 y-dir
    plt.plot(F_tx, 'g--', marker='o', markersize=5, markevery=10, linewidth=2, label="Prediction x-dir")  # 预测结果 x-dir
    plt.plot(F_ty, 'r--', marker='o', markersize=5, markevery=10, linewidth=2, label="Prediction y-dir")  # 预测结果 y-dir

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("$D_v$", fontsize=20, fontstyle='italic', labelpad=20)

    
    plt.subplots_adjust(bottom=0.2)  # 增加底部边距

    plt.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=False, fancybox=False)

    plt.subplots_adjust(hspace=0.1)  # 调整子图之间的间距
    
    plt.savefig(path+f'/Re(%2e)_'%Re+'Pressure Force.png')
    plt.close()

    mean_u = torch.from_numpy(boundary_zone['mean_u'])
    boundary_pos = torch.from_numpy(boundary_zone['mesh_pos'][0])
    # characteristic_length = np.sqrt((boundary_zone['mesh_pos'][0][:,0].max() - boundary_zone['mesh_pos'][0][:,0].min())**2+(boundary_zone['mesh_pos'][0][:,1].max() - boundary_zone['mesh_pos'][0][:,1].min())**2)
    
    _,boundaryleft_index = torch.max(boundary_pos[:,0:1],dim=0)
    _,boundaryright_index = torch.min(boundary_pos[:,0:1],dim=0)
    
    characteristic_length = torch.norm(boundary_pos[boundaryleft_index]-boundary_pos[boundaryright_index]).numpy()
    
    cylinder_D = characteristic_length

    F_D = F_px + F_tx
    F_L = F_py + F_ty
    taget_F_D = taget_F_px + taget_F_tx
    taget_F_L = taget_F_py + taget_F_ty

    Cd = 2 * F_D / (1 * mean_u**2 * cylinder_D)
    Cl = 2 * F_L / (1 * mean_u**2 * cylinder_D)
    target_C_D = 2 * taget_F_D / (1 * mean_u**2 * cylinder_D)
    target_C_L = 2 * taget_F_L / (1 * mean_u**2 * cylinder_D)
    
    CD_CL_dict={'Cd':Cd.numpy().tolist(),
                'Cl':Cl.numpy().tolist(),
                'target_C_D':target_C_D.numpy().tolist(),
                'target_C_L':target_C_L.numpy().tolist(),
    }
    with open(path + f'/Re(%2e)_' % Re + 'Lift_and_Drag_factor.json', "w") as file:
        json.dump(CD_CL_dict, file, ensure_ascii=False, indent=4)
        
    # 修改时间步范围为200-599
    plot_start_time_steps = 200
    plot_end_time_steps = Cd.size(0)-1
    
    RMSE_CD = (torch.sum((target_C_D[plot_start_time_steps:plot_end_time_steps]-Cd[plot_start_time_steps:plot_end_time_steps])**2)/torch.sum(Cd[plot_start_time_steps:plot_end_time_steps]**2)).numpy()
    RMSE_CL = (torch.sum((target_C_L[plot_start_time_steps:plot_end_time_steps]-Cl[plot_start_time_steps:plot_end_time_steps])**2)/torch.sum(Cl[plot_start_time_steps:plot_end_time_steps]**2)).numpy()
    
    with open(path + f'/Re(%2e)_' % Re + f'steps_{plot_start_time_steps}_{plot_end_time_steps}_Lift_and_Drag_factorRMSE.txt', "w") as f_txt:
        f_txt.write(f'RMSE_CD:{RMSE_CD}\n')
        f_txt.write(f'RMSE_CL:{RMSE_CL}\n')
    print(f'RMSE_CD:{RMSE_CD},RMSE_CL:{RMSE_CL}')

    plot_CD_max = torch.maximum(Cd[plot_start_time_steps:plot_end_time_steps].max(), target_C_D[plot_start_time_steps:plot_end_time_steps].max())
    
    if plot_CD_max > 0:
        plot_CD_max = 1.1*plot_CD_max  # Set y-axis limits to center the curve
    else:
        plot_CD_max = 0.9*plot_CD_max  # Set y-axis limits to center the curve
    
    plot_CL_max = torch.maximum(Cl[plot_start_time_steps:plot_end_time_steps].max(), target_C_L[plot_start_time_steps:plot_end_time_steps].max())
    if plot_CL_max > 0:
        plot_CL_max = 1.5*plot_CL_max  # Set y-axis limits to center the curve
    else:
        plot_CL_max = 0.5*plot_CL_max  # Set y-axis limits to center the curve
        
    plot_CD_min = torch.minimum(Cd[plot_start_time_steps:plot_end_time_steps].min(), target_C_D[plot_start_time_steps:plot_end_time_steps].min())
    if plot_CD_min > 0:
        plot_CD_min = 0.9*plot_CD_min  # Set y-axis limits to center the curve
    else:
        plot_CD_min = 1.1*plot_CD_min  # Set y-axis limits to center the curve
        
    plot_CL_min = torch.minimum(Cl[plot_start_time_steps:plot_end_time_steps].min(), target_C_L[plot_start_time_steps:plot_end_time_steps].min())
    if plot_CL_min > 0:
        plot_CL_min = 0.5*plot_CL_min  # Set y-axis limits to center the curve
    else:
        plot_CL_min = 1.5*plot_CL_min  # Set y-axis limits to center the curve

    plt.figure(figsize=(16, 7))
    plot_ranges = [(plot_start_time_steps, 400), (400, Cd.size(0))]
    
    for i, (start_plot, end_plot) in enumerate(plot_ranges):
        
        smooth_CD = Cd[start_plot:end_plot]
        smooth_CL = Cl[start_plot:end_plot]
        smooth_target_CD = target_C_D[start_plot:end_plot]
        smooth_target_CL = target_C_L[start_plot:end_plot]
        # 绘制 Cd
        ax_cd = plt.subplot(2, 2, i+1)
        ax_cd.set_xticklabels([])
        ax_cd.spines['right'].set_visible(False)
        ax_cd.spines['top'].set_visible(False)
        ax_cd.set_xlim(start_plot, end_plot)
        
        # 设置Y轴的主要刻度定位器为6个刻度
        ax_cd.yaxis.set_major_locator(MaxNLocator(5))
        
        # 添加X轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
        ax_cd.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        # 添加Y轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
        ax_cd.yaxis.set_minor_locator(AutoMinorLocator(4))
        
        ax_cd.tick_params(which='both', direction='in')
        
        plt.plot(range(start_plot, end_plot), smooth_target_CD, 'k-', linewidth=2, label="CFD")
        plt.plot(range(start_plot, end_plot), smooth_CD, 'r--', marker='o', markersize=4,markevery=8, label="Prediction")
        
        if i==0:
            plt.ylabel("$C_d$", fontsize=20, fontstyle='italic')
        else:
            plt.legend(fontsize=15, loc='upper right')
        plt.yticks(fontsize=10)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().set_ylim(plot_CD_min, plot_CD_max)
        ax_cd.set_xticklabels([])

        # 绘制 Cl
        ax_cl = plt.subplot(2, 2, i+3)
        ax_cl.spines['right'].set_visible(False)
        ax_cl.spines['top'].set_visible(False)
        ax_cl.set_xlim(start_plot, end_plot)
        
        # 设置Y轴的主要刻度定位器为6个刻度
        ax_cl.yaxis.set_major_locator(MaxNLocator(5))
        
        # 添加X轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
        ax_cl.xaxis.set_minor_locator(AutoMinorLocator(4))
        
        # 添加Y轴的次要刻度，这里设置为每个主要刻度之间有4个次要刻度
        ax_cl.yaxis.set_minor_locator(AutoMinorLocator(4))
        
        ax_cl.tick_params(which='both', direction='in')
        # # 启用网格线显示主要刻度线
        # ax_cl.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        
        # # 启用网格线显示次要刻度线，更细且为虚线
        # ax_cl.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        
        plt.plot(range(start_plot, end_plot), smooth_target_CL, 'k-', linewidth=2, label="CFD")
        plt.plot(range(start_plot, end_plot), smooth_CL, 'r--', marker='o', markersize=4, markevery=2, label="Prediction")
        

        plt.xlabel("Time steps", fontsize=12)
        if i==0:
            plt.ylabel("$C_l$", fontsize=20, fontstyle='italic')
        else:
            plt.legend(fontsize=15, loc='upper right')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().set_ylim(plot_CL_min, plot_CL_max)

        plt.subplots_adjust(hspace=0.1,wspace=0.1)  # 调整子图之间的间距
        
    # 保存图像
    plt.savefig(os.path.join(os.path.split(save_path)[0], f'Re({Re:2e})_Lift_and_Drag_coefficient.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return True
