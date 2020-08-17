def create_labels():
	with open("../data/file_list.txt", 'r') as f:
		videos = f.readlines()
	
	with open("../data/labels.txt", "w+") as labels:
		for video in videos:
			v = video.split(".")[0]
			label = 0 if video.split("_")[1]=="truth" else 1
			labels.write(v + " " + str(label) + "\n")


if __name__ == '__main__':
	create_labels()