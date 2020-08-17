import torch
from torch.autograd import Variable
from torch.autograd import Function
import cv2
import sys
import numpy as np
import argparse
from utils import get_dataset, load_checkpoint
from train import get_model
from torch.utils.data import DataLoader
import pdb
from gradcam_lkp_net_video import GradCAM
import utils
import matplotlib.pyplot as plt


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
		test_list = [l.strip() for l in f.readlines()]
	dl   = DataLoader(get_dataset(args, test_list), shuffle=False, num_workers=0, batch_size=1)
	for data_batch, labels_batch in dl:
		# if args.use_gpu:
		# 	data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
		
		target_index = None
		input_var = torch.autograd.Variable(data_batch, volatile=True).cuda()
		gcam.forward(input_var)
		gcam.prob, gcam.idx = gcam.probs.data.squeeze().sort(0, True)
		gcam.backward(idx=[1])
		gcam_map = gcam.generate(args.gradcam_layer, 'raw')
		gcam_map = gcam_map.cpu()

		mask = torch.nn.Upsample(scale_factor=5, mode='nearest')(gcam_map.view(1,1,36))
		mask = mask.detach().cpu().numpy().squeeze()
		mask = (mask-mask.min()) / (mask.max()-mask.min())
		for k in range(17):
			x = data_batch[0, 2+k, :].cpu().numpy()
			plt.plot(range(180), x, label=f'au-{k}')
		plt.plot(range(180), mask, label='attn')
		plt.legend()
		plt.show()
		# pdb.set_trace()
