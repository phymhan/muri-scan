import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

directory = ['deceptive', 'truthfull']
names = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r',
		 'AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r'
		]

def viz_durations(path):
	frames_num  = []
	frames_dict = {}
	for filename in os.listdir(path):
		data = np.load(path+filename)
		frames = np.array(data['arr_0'][:, 15:], dtype='float32').shape[0]
		frames_num.append(frames)
		if frames not in frames_dict.keys():
			frames_dict[frames] = []
		frames_dict[frames].append(filename)

	print(min(frames_num), max(frames_num))
	frames_num.sort()
	print(frames_num[:50])

	_ = plt.hist(frames_num, bins='auto') 		# arguments are passed to np.histogram
	plt.title("Histogram with 'auto' bins")
	plt.show()

def viz_waveforms(path):
	save_path = "../data/openface/figs/"
	with open("../data/labels.txt", 'r') as f:
		lf = f.readlines()
		labels = {'-'.join((l.split()[0]).split('/')): int(l.split()[1]) for l in lf}
	for filename in os.listdir(path):
		data = np.load(path+filename)
		features = np.array(data['arr_0'][:, 15:], dtype='float32').T
		print(filename, labels[filename.split(".")[0]], features.shape)
		fig = plt.figure()
		for i in range(9):
			plt.subplot(3,3,i+1)
			plt.plot(list(range(features.shape[1])), features[i])
			plt.title(names[i])
		fig = plt.gcf()
		fig.set_size_inches((12, 12), forward=False)
		fig.savefig(os.path.join(save_path, directory[labels[filename.split(".")[0]]], filename.split(".")[0]+"-1.png"))
		plt.clf()
		fig = plt.figure()
		for i in range(9,17):
			plt.subplot(2,4,i-8)
			plt.plot(list(range(features.shape[1])), features[i])
			plt.title(names[i])
		fig = plt.gcf()
		fig.set_size_inches((15,9), forward=False)
		fig.savefig(os.path.join(save_path, directory[labels[filename.split(".")[0]]], filename.split(".")[0]+"-2.png"), dpi=500)
		plt.clf()
		
	

if __name__ == '__main__':
	
	path        = "../data/openface/npz-use/"
	# viz_durations(path)	
	viz_waveforms(path)
