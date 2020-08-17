import numpy as np


def partition(list_in, n):
    np.random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def check_folds(folds, video_dict):
	a = []
	for f in folds:
		s = 0
		for name in f:
			s += len(video_dict[name])
		a.append(s)
	#print(folds)
	print(a)


def create_folds_ids10(video_dict):

	np.random.seed(10)
	id_list = list(video_dict.keys())
	folds = partition(id_list, 10)
	check_folds(folds, video_dict)

	hh = []
	for i, fold in enumerate(folds):
		for name in fold:
			for sample in video_dict[name]:
				hh.append(sample + " " + str(i))

	with open("../data/splits_10.txt", 'w') as f:
		f.write("\n".join(hh))

def create_folds_ids3(video_dict):
	
	np.random.seed(21)
	id_list = list(video_dict.keys())
	folds = partition(id_list, 3)
	check_folds(folds, video_dict)

	hh = []
	for i, fold in enumerate(folds):
		for name in fold:
			for sample in video_dict[name]:
				hh.append(sample + " " + str(i))

	with open("../data/splits_3.txt", 'w') as f:
		f.write("\n".join(hh))

	
def find_best_fold(list_in, n):
	best_seed = 0
	best_diff = 100
	folds = []
	for i in range(1000):
		np.random.seed(i)
		folds.append(partition(list_in, n))
		lies  = []
		truth = []
		for p in folds[i]:
			lies.append(sum(x[0] for x in p))
			truth.append(sum(x[1] for x in p))
		diff = sum(abs(lies[j]-truth[j]) for j in range(n)) 
		if diff < best_diff:
			best_seed = i
			best_diff = diff
			print(i, lies, truth)
			
	return best_seed, folds[best_seed]


def find_best_fold_ids(dict_in, n):
	best_seed = 0
	best_diff = 100
	folds = []
	name_list = list(video_dict)
	for i in range(1000):
		np.random.seed(i)
		folds.append(partition(name_list, n))
		lies  = []
		truth = []
		for fold in folds[i]:
			cnt_lie = 0
			cnt_tru = 0
			for name in fold:
				cnt_lie += sum(it[1] for it in video_dict[name])
				cnt_tru += sum(it[0] for it in video_dict[name])
			lies.append(cnt_lie)
			truth.append(cnt_tru)
		diff = sum(abs(lies[j]-truth[j]) for j in range(n)) 
		if diff < best_diff:
			best_seed = i
			best_diff = diff
			print(i, lies, truth)
			
	return best_seed, folds[best_seed]


def create_folds(video_dict, num):
	dd = {'lie': 1, 'truth': 0}
	for key in video_dict.keys():
		for i in range(len(video_dict[key])):
			video_dict[key][i] = [1,0,video_dict[key][i]] if "lie" in video_dict[key][i] else [0,1,video_dict[key][i]] 
	
	seed, folds = find_best_fold_ids(video_dict, num)
	print(folds)
	
	ll = []
	for i, fold in enumerate(folds):
		for name in fold:
			for video in video_dict[name]:
				ll.append(video[2] +" "+ str(i))
	
	with open("../data/splits-ids_{}.txt".format(num), 'w') as f:
		f.write("\n".join(ll))



if __name__ == '__main__':

	with open("../data/file_list.txt", 'r') as f:
		video_list = list(map(lambda x: x.rstrip().split("-"), f.readlines()))
	video_dict = {}
	for video in video_list:
		v = video[0].split(".")[0]
		if video[1] in video_dict:
			video_dict[video[1]].append(v)
		else:
			video_dict[video[1]] = [v]
	
	# create_folds_ids10(video_dict)
	# create_folds_ids3(video_dict)
	create_folds(video_dict, 10)
