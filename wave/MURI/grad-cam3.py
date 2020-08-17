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


vidroot2 = '/home/lh599/Active/muri-scan/data/npz-video'
outroot = 'data/attn_vis'

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
		# vidpath1 = os.path.join(vidroot1, f'{filename}.mp4')
		# vidpath2 = os.path.join(vidroot2, filename, f'{filename}.avi')
		csvpath = os.path.join(vidroot2, f'{filename}.npz')
		outpath = os.path.join(outroot, f'{filename}')

		# X = read_csv(csvpath)
		X = np.load(csvpath)['features']
		X_gaze = X[:, 6:8]
		X_au   = X[:, 14:]
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
