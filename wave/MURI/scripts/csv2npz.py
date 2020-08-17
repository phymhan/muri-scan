import os
import shutil
import numpy as np
import pandas as pd


keep_cols = [' confidence',' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z',' gaze_angle_x',' gaze_angle_y',
			 ' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz',
			 ' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',
			 ' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r'
			]

def copy_csv(video_dict):
	dicts = ["../data/openface/Deceptive/", "../data/openface/Truthful/"]
	for d in dicts:
		for video in video_dict.keys():
			if video in os.listdir(d):
				shutil.copyfile(os.path.join(d, video, video+".csv"), "../data/openface/csv/"+video+".csv")


def create_npz_old(video_dict):
	dicts = ["../data/openface/Deceptive/", "../data/openface/Truthful/"]
	for d in dicts:
		for video in video_dict.keys():
			if video in os.listdir(d):
				df = pd.read_csv(os.path.join(d, video, video+".csv"))

				X = df.as_matrix(columns=keep_cols)
				if len(video_dict[video])>1:	
					if len(video_dict[video])==2:
						if "start" in video_dict[video][1]:
							start = int(video_dict[video][1].split("=")[1])
							end = 1e5
						else:
							start = 0
							end = int(video_dict[video][1].split("=")[1])

					elif len(video_dict[video])==3:
						start, end = int(video_dict[video][1].split("=")[1]), int(video_dict[video][2].split("=")[1])
					
					l = []
					m = []
					if start>0:
						l = list(range(start*30))
					if end*30 < X.shape[0]:
						m = list(range(end*30, X.shape[0]))	
					Y = np.delete(X, l+m, axis=0)
					np.savez('../data/openface/npz-use/'+video, Y)
				else:
					np.savez('../data/openface/npz-use/'+video, X)


def create_csv(video_dict):
	d = "../data/openface/csv"
	for video in video_dict.keys():
		df = pd.read_csv(os.path.join(d, video+".csv"))
		X = df.as_matrix(columns=keep_cols)
		if len(video_dict[video])>1:	
			if len(video_dict[video])==2:
				if "start" in video_dict[video][1]:
					start = int(video_dict[video][1].split("=")[1])
					end = 1e5
				else:
					start = 0
					end = int(video_dict[video][1].split("=")[1])

			elif len(video_dict[video])==3:
				start, end = int(video_dict[video][1].split("=")[1]), int(video_dict[video][2].split("=")[1])
			
			l = []
			m = []
			if start>0:
				l = list(range(start*30))
			if end*30 < X.shape[0]:
				m = list(range(end*30, X.shape[0]))	
			Y = np.delete(X, l+m, axis=0)
			np.savez('../data/openface/npz-use/'+video, Y)
		else:
			np.savez('../data/openface/npz-use/'+video, X)


if __name__ == '__main__':
	
	with open("../data/file_list.txt", 'r') as f:
		video_list = list(map(lambda x: x.rstrip().split("-"), f.readlines()))
	video_dict = {video[0].split(".")[0] : video[1:] for video in video_list}
	
	# id_dict = {video[1] : video[0].split(".") for video in video_list}
	# print(len(id_dict))
	# copy_csv(video_dict)
	
	create_csv(video_dict)