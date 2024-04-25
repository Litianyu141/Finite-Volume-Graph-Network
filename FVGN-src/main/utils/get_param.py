import argparse
from xmlrpc.client import boolean
import json
import os
def str2bool(v):
	"""
	'boolean type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def params(load=None,fore_args_parser=None):
	if load is not None:
		if fore_args_parser is not None:
			parser = fore_args_parser
			params = vars(parser.parse_args())
   
			with open(load+'/commandline_args.json', 'rt') as f:
				params.update(json.load(f))
    
			for k, v in params.items():
				try:
					parser.add_argument('--' + k, default=v)
				except:
					pass
    
			params = parser.parse_args()
   
			return params

		else:
			parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')
			params = vars(parser.parse_args())
   
			with open(load+'/commandline_args.json', 'rt') as f:
				params.update(json.load(f))
			for k, v in params.items():
				parser.add_argument('--' + k, default=v)
			args = parser.parse_args()
			return  args
	else:
		"""
		return parameters for training / testing / plotting of models
		:return: parameter-Namespace
		"""
		parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')

		''' >>> Training parameters >>> '''
		parser.add_argument('--net', default="GN-Cell", type=str, help='network to train (default: GN-Cell)', choices=["GN-Cell","GN-Node"])
		parser.add_argument('--n_epochs', default=14000, type=int, help='number of epochs (after each epoch, the model gets saved)')
		parser.add_argument('--hidden_size', default=128, type=int, help='hidden size of network (default: 20)')
		parser.add_argument('--traj_length', default=600, type=int, help='dataset traj_length (default: cylinder 599)')
		parser.add_argument('--batch_size', default=16, type=int, help='batch size (default: 100)')
		parser.add_argument('--cumulative_length', default=600, type=int, help='number of time steps to accmulate stastics in normalizer (default: 600)')
		parser.add_argument('--train_traj_length', default=400, type=int, help='number of time steps to train  (default: 300)')
		parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
		parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
  
		# loss params
		parser.add_argument('--loss_cont', default=0, type=float, help='loss factor for continuity equation')
		parser.add_argument('--loss_mom', default=10, type=float, help='loss factor for uv diffusion flux on face')
		parser.add_argument('--loss_face_uv', default=1, type=float, help='loss factor for uvp flux on face')
		parser.add_argument('--loss_face_p', default=1, type=float, help='regularizer for gradient of p. evt needed for very high reynolds numbers (default: 0)')
		parser.add_argument('--loss', default='global_mean_sum', type=str, help='loss type to train network (default: square)',choices=['global_mean_sum','global_mean_mean','direct_mean_loss'])
  
		# nn config params
		parser.add_argument('--drop_out', default=False, type=str2bool, help='using dropout technique in message passing layer(default: True)')
		parser.add_argument('--mp_times', default=2, type=int, help='message passing times(default: True)')
		parser.add_argument('--multihead', default=1, type=int, help='using dropout technique in message passing layer(default: True)')
		parser.add_argument('--dual_edge', default=False, type=str2bool, help='whether use dual edge direction encoding in message passing layer(default: True)')
		parser.add_argument('--pre_statistics_times', default=2, type=int, help='accumlate data statistics for normalization before backprapagation (default: 1)')
		parser.add_argument('--Noise_injection_factor', default=2e-2, type=float, help='factor for normal Noise distrubation,0 means No using Noise injection ,(default: 2e-2),choices=["0","Greater than 0"]')
		parser.add_argument('--nn_act', default="SiLU", type=str, help='what activate function to use ,(default: "SiLU")',choices=["SiLU","ReLU"])
  
		# lr params
		parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of optimizer (default: 0.0001)')
		parser.add_argument('--lr_milestone', default=3000, type=int, help='when to downgrade learning rate at certain epoch  (default: 3000)')
		parser.add_argument('--lr_expgamma', default=1e-2, type=float, help='how much to downgrade learning rate at certain epoch in exp  (default: 7500)')
		parser.add_argument('--lr_milestone_gamma', default=0.1, type=float, help='how much  to downgrade learning rate at certain epoch with certain gamma  (default: 3000)')
		parser.add_argument('--before_explr_decay_steps', default=7000, type=int, help='steps before using exp lr decay technique (default:12000)')
		''' <<< Training parameters <<< '''
  
		# Fluid parameters
		parser.add_argument('--rho', default=1, type=float, help='fluid density rho')
		parser.add_argument('--mu', default=0.001, type=float, help='fluid viscosity mu')
		parser.add_argument('--dt', default=0.01, type=float, help='timestep of fluid integrator')
		
		# Load parameters
		parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
		parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
		parser.add_argument('--start_over', default=False, type=str2bool, help='whether start epoch from zero, can be used after load a state (default: True)')
		parser.add_argument('--load_optimizer', default=True, type=str2bool, help='load state of optimizer (default: True)')
		parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
		
		# model parameters
		parser.add_argument('--message_passing_num', default=15, type=int, help='message passing layer number (default:15)')
		parser.add_argument('--cell_input_size', default=2, type=int, help='cell encoder cell_input_size: [u,v] (default: 2)')
		parser.add_argument('--cell_one_hot', default=0, type=int, help='cell one hot dimention (default: 0)')
		parser.add_argument('--edge_input_size', default=5, type=int, help='edge encoder edge_input_size: [u,v,relative mesh_pos,and edge_length] (default: 3)')
		parser.add_argument('--edge_one_hot', default=9, type=int, help='edge one hot dimention (default: 9)')
		parser.add_argument('--cell_target_input_size', default=2, type=int, help='cell normlizer cell_target_input_size [u_cell_target,v_cell_target](default: 2)')
		parser.add_argument('--face_flux_target_input_size', default=3, type=int, help='face_flux_target_input_size include [uf,vf,pf] (default: 3)')
		parser.add_argument('--edge_output_size', default=5, type=int, help='edge decoder edge_output_size, [uf,vf,pf,cx,cy] (default: 8)')
		parser.add_argument('--cell_output_size', default=2, type=int, help='cell decoder cell_output_size (default: 1)')
		parser.add_argument('--grid_feature_size', default=1, type=int, help='face_length_normlizer input_size (default: 1)')

		# dataset params
		parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
		parser.add_argument('--dataset_type', default="h5", type=str, help='define your dataset file type tfrecode or h5 (default:tf)')
		parser.add_argument('--dataset_dir', default='/home/litianyu/dataset/MeshGN/cylinder_flow/mesh_with_target_on_node_chk', type=str, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
		parser.add_argument('--dataset_dir_h5', default='/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset/conveted_h5', type=str, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
  
		# validate params
		parser.add_argument('--model_dir', default='', type=str, help='load latest version for validating (if True: leave load_date_time and load_index None. default: False)')
  
		#git information
		if False:
			import git
			currentdir = os.getcwd()
			repo = git.Repo(currentdir)
			current_branch=repo.active_branch.name
			commits = list(repo.iter_commits(current_branch, max_count=5))   
			parser.add_argument('--git_branch', default=current_branch, type=str, help='current running code`s git branch')
			parser.add_argument('--git_messages', default=commits[0].message, type=str, help='current running code`s git messages')
			parser.add_argument('--git_commit_dates', default=str(commits[0].authored_datetime), type=str, help='current running code`s git commit date')
			params = parser.parse_args()
			
			# git information
			git_info = {"git_branch":params.git_branch,
            "git_commit_dates":params.git_commit_dates}
			# parse parameters
			
			return params,git_info
		else:
			parser.add_argument('--git_branch', default="FVGN-NI", type=str, help='current running code git branch')
			parser.add_argument('--git_commit_dates', default="", type=str, help='current running code git commit date')	
			params = parser.parse_args()
			git_info = {"git_branch":params.git_branch,
            "git_commit_dates":params.git_commit_dates}
			return params,git_info

def get_hyperparam(params):
	return f"net {params.net}; hs {params.hidden_size}; mu {params.mu}; rho {params.rho}; dt {params.dt};"
