import torch
from torch.autograd import Variable
from torch.autograd import Function
import cv2
import sys
import numpy as np
import numpy.matlib
import argparse
from utils import get_dataset, load_checkpoint
from train import get_model
from torch.utils.data import DataLoader
import pdb
from gradcam_lkp_net_video import GradCAM
import utils
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import animation


vidroot1 = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/all_clips'
vidroot2 = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/all_openface'
outroot = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/attn_vis'

keep_cols = [' confidence',' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z',' gaze_angle_x',' gaze_angle_y',
			 ' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz',
			 ' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',
			 ' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r'
			]

def read_csv(filename):
	df = pd.read_csv(filename)
	X = df.as_matrix(columns=keep_cols)
	return X

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	# args = get_args()
	args = utils.Options().get_options()

	model = get_model(args)
	model.train()
	load_checkpoint('experiments/checkpoints/best.pth.tar', model)

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	gcam = GradCAM(model=model, target_layers=[args.gradcam_layer], n_class=2, cuda=True)
	with open(args.test_list, 'r') as f:
		test_list = [l.strip().split()[0] for l in f.readlines()]
	
	for filename in test_list:
		vidpath1 = os.path.join(vidroot1, f'{filename}.mp4')
		vidpath2 = os.path.join(vidroot2, filename, f'{filename}.avi')
		csvpath = os.path.join(vidroot2, filename, f'{filename}.csv')
		outpath = os.path.join(outroot, f'{filename}')
		# if not os.path.exists(outpath):
		# 	os.makedirs(outpath)
		X = read_csv(csvpath)
		X_gaze = X[:, 7:9]
		X_au   = X[:, 15:]
		feat = np.concatenate((X_gaze, X_au), axis=1)
		feat = torch.FloatTensor(feat.T).view(1, 19, -1)
		input_var = torch.autograd.Variable(feat, volatile=True).cuda()
		gcam.forward(input_var)
		gcam.prob, gcam.idx = gcam.probs.data.squeeze().sort(0, True)
		gcam.backward(idx=[1])
		gcam_map = gcam.generate(args.gradcam_layer, 'raw')
		gcam_map = gcam_map.cpu()

		# pdb.set_trace()

		mask = torch.nn.Upsample(scale_factor=5, mode='nearest')(gcam_map.view(1,1,-1))
		mask = mask.detach().cpu().numpy().squeeze()
		# mask = (mask-mask.min()) / (mask.max()-mask.min())
		
		print(filename)
		feat = feat[0, 2:, :].cpu().numpy()
		np.save(os.path.join(outroot, f'{filename}_au.npy'), feat)
		np.save(os.path.join(outroot, f'{filename}_att.npy'), mask)


		# for k in range(17):
		# 	x = feat[0, 2+k, :].cpu().numpy()
		# 	plt.plot(x, label=f'au-{k}')
		# plt.plot(mask, label='attn', linewidth=3)
		# plt.legend()
		# plt.show()

		# pdb.set_trace()

		# vid1 = cv2.VideoCapture(vidpath1)
		# vid2 = cv2.VideoCapture(vidpath2)
		# num_frame = mask.shape[0]

		# fig = plt.figure()
		# ax = plt.axes(xlim=(0, num_frame), ylim=(0, 2))
		# line, = ax.plot([], [], lw=2)
		# feat = feat[0, 2:, :].cpu().numpy()

		# pdb.set_trace()

		# def init():
		# 	line.set_data([], [])
		# 	return line,
		
		# def animate(i):
		# 	x = np.matlib.repmat(range(i), 17, 1)
		# 	y = feat[:, :i]
		# 	line.set_data(x.T, y.T)
		# 	return line,
		
		# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frame,
		# 							   blit=True)
		# anim.save('test_anim.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
		# plt.show()

		# fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 3]})
		# ax[0].set_axis_off()
		# ax[1].set_axis_off()

		# frames = []
		# for i in range(30):
		# 	_, im1 = vid1.read()
		# 	_, im2 = vid2.read()
		# 	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
		# 	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
		# 	frames.append([
		# 		ax[0].imshow(im1, animated=True, aspect='equal'),
		# 		ax[1].imshow(im2, animated=True, aspect='equal'),
				
		# 		])
		# 	# pdb.set_trace()
		# ani = animation.ArtistAnimation(fig, frames)

		# plt.show()
		# vid1.release()
		# vid2.release()
