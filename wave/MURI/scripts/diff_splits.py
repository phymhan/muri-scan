import os

if __name__ == '__main__':
	
	with open("../data/file_list.txt", 'r') as f:
		video_list = list(map(lambda x: x.rstrip().split("-"), f.readlines()))
	set1 = set([video[0].split(".")[0] for video in video_list])
	with open("../data/video_list.txt", 'r') as f:
		video_list = f.readlines()
	set2 = set([video.split(".")[0] for video in video_list])
	set3 = set(list(map( lambda x: x.split(".")[0] , os.listdir("../data/clips/Truthful") + os.listdir("../data/clips/Deceptive"))))

	print(set2 - set1)
	# print("++++")
	# l = list(set3 -set2)
	# l.sort()
	# print(l)