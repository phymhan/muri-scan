
import os
import numpy as np

cols = [f'X_{i}' for i in range(17,68)] + [f'Y_{i}' for i in range(17,68)] + [f'Z_{i}' for i in range(17,68)]
global_fs = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
			 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
			  'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 
			  'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

with open('../data/openface/AZ-005_1_R3.csv', 'r') as f:
    h = [hh.strip(' ') for hh in f.readline().rstrip('\n').split(',')]
col_idx = [h.index(hh) for hh in cols]
glob_idx = [h.index(hh) for hh in global_fs]

with open('../data/filelist.txt', 'r') as f:
	files = [i.rstrip('\n') for i in f.readlines()]

for sample in files:
	if not os.path.exists('../data/npz/' + '-'.join(sample.split()) + '.npz'):
		landmarks_list = []
		landmarks = np.genfromtxt('../data/openface/'+sample.split()[0] + '.csv', delimiter=',')[1:,col_idx]
		features = np.genfromtxt('../data/openface/'+sample.split()[0] + '.csv', delimiter=',')[1:, glob_idx]
		for i in range(int(sample.split()[1]), int(sample.split()[1]) + 900):
			land = landmarks[i]
			landmarks_list.append(land.reshape((3, 68-17)).T)
		features = features[int(sample.split()[1]): int(sample.split()[1]) + 900,...]

		np.savez('../data/npz/' + '-'.join(sample.split()) + '.npz', landmarks=landmarks_list, features=features)