import os

#path = '/home/anastasis/Desktop/videos/'
path = '/dresden/users/as2947/MURI/clips-r3/'

for c in os.listdir(path):
	clip_name = path + c
	fields    = clip_name.split("_")
	num 	  = int(fields[1])
	if num < 10:
		new_name = fields[0] + "-00" + str(num) + "_" + fields[2] + "_" + fields[3] + fields[4]
	elif num < 100:
		new_name = fields[0] + "-0" + str(num) + "_" + fields[2] + "_" + fields[3] + fields[4]
	else:
		new_name = fields[0] + "-" + str(num) + "_" + fields[2] + "_" + fields[3] + fields[4]
	os.rename(clip_name, new_name)
	