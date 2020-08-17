import os
import numpy as np
import pandas as pd


keep_cols = [' confidence',' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z',' gaze_angle_x',' gaze_angle_y',
			 ' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz',
			 ' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',
			 ' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r'
			]
dicts = {"lie" : "../data/openface/Deceptive/", "truth" : "../data/openface/Truthful/"}


def interpolate1():
	# Interpolate files tha contain videos with 24, 25 and 30 fps
	freqs = {27: 10, 25: 6, 24: 5}
	fps_dict = { 24: ['trial_truth_021'], 
				 25: ['trial_truth_018', 'trial_truth_019', 'trial_truth_020', 'trial_truth_030', 'trial_truth_037', 'trial_lie_033'],
				 27: ['trial_truth_053', 'trial_truth_060', 'trial_lie_037', 'trial_lie_038', 'trial_lie_039', 'trial_lie_040']
				}
	for fps in fps_dict.keys():
		for video in fps_dict[fps]:
			categ = video.split("_")[1]
			dic  = dicts[categ]
			freq = freqs[fps]
			print(video)
			df = pd.read_csv(os.path.join(dic, video, video+".csv"))[keep_cols]
			X = df.as_matrix()
			Y = -1*np.ones((30*df.shape[0]//fps, df.shape[1]), dtype=np.float32)
			# copy X to Y leaving some space for values to be interpolated
			row = 0
			for i in range(Y.shape[0]):
				if (i+1)%freq!=0:
					Y[i] = X[row]
					row += 1
			
			for i in range(Y.shape[0]):
				if (i+1)%freq==0:
					if i+1==Y.shape[0]:
						Y[i] = Y[i-1]
					else:
						Y[i] = (Y[i-1] + Y[i+1])/2

			with open("../data/openface/new/"+video+".csv", 'w') as f:
				f.write(','.join(keep_cols) + '\n')
				for i in range(Y.shape[0]):
					ll = [str(w) for w in Y[i]]
					f.write(','.join(ll) + '\n')

def interpolate2():
	# Interpolate files tha contain videos with 10 fps
	fps10_list = ['trial_truth_0{}'.format(i) for i in range(43,51)] + ['trial_truth_052', 'trial_lie_035']
	for video in fps10_list:
		categ = video.split("_")[1]
		dic  = dicts[categ]
		print(video)
		df = pd.read_csv(os.path.join(dic, video, video+".csv"))[keep_cols]
		X = df.as_matrix()
		Y = -1*np.ones((3*df.shape[0]-2, df.shape[1]), dtype=np.float32)
		# copy X to Y leaving some space for values to be interpolated
		row = 0
		for i in range(Y.shape[0]):
			if i%3==0:
				Y[i] = X[row]
				row += 1
		
		for i in range(Y.shape[0]):
			if i%3==1:
				if i+2<Y.shape[0]:
					Y[i] = (Y[i-1] + Y[i+2])/2
					Y[i+1] = Y[i]
				else:
					Y[i] = Y[i-1]
					Y[i+1] = Y[i]

		with open("../data/openface/new/"+video+".csv", 'w') as f:
			f.write(','.join(keep_cols) + '\n')
			for i in range(Y.shape[0]):
				ll = [str(w) for w in Y[i]]
				f.write(','.join(ll) + '\n')



if __name__ == '__main__':

	interpolate1()
	interpolate2()
