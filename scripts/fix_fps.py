import os
import subprocess
from moviepy.editor import *

#path = '/home/anastasis/Desktop/videos/'
path = '/dresden/users/as2947/MURI/clips-r3/'
not_use_list = []
for c in os.listdir(path):
	clip_name = path + c
	fps  = int(subprocess.check_output("ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {}".format(clip_name),
	    				                shell=True).decode("utf-8").split("/")[0])
	if fps != 30:
		clip = VideoFileClip(clip_name)
		clip.write_videofile(clip_name, fps=30)