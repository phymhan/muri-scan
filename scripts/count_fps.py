import os
import sys
import subprocess

#path = '/home/anastasis/Desktop/'
path = '/dresden/users/as2947/MURI/clips-r3/'
not_use_list = []
for clip in os.listdir(path):
    fps = int(subprocess.check_output("ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {}".format(path+clip),
                                 shell=True).decode("utf-8").split("/")[0])
    if fps != 30:
        not_use_list.append(clip)
        print("Video : {} --- FPS : {}".format(clip, fps))

print("Number of problematic files : {}".format(len(not_use_list)))
