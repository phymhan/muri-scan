import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
	
	keep_cols = [' confidence',' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z',' gaze_angle_x',' gaze_angle_y',
				 ' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz',
				 ' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',
				 ' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r'
				]
	dic = "../data/openface/"
	df  = pd.read_csv('../data/annotations_fixed.csv')
	for i in range(df.shape[0]):
		user, run = df['video'][i].split("/")[3:5]
		start = df['start'][i]
		end   = df['end'][i]		
		dframe = pd.read_csv(os.path.join(dic,user,run,"video.csv"))
		if dframe[' face_id'].any():
			print("There was a problem in tracking video: {}".format(user + run))
		X = dframe.as_matrix(columns=keep_cols)
		l = []
		m = []
		if start>0:
			l = list(range(start*30))
		if end*30 < X.shape[0]:
			m = list(range(end*30, X.shape[0]))	
		Y = np.delete(X, l+m, axis=0)
		# np.savez(dic+'npz/'+user+'-'+run, X)
		np.savez(dic+'npz-use/'+user+'-'+run, Y)
		