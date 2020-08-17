
import numpy as np
import pandas as pd


def partition(list_in, n):
    np.random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def search_random_seed(list_in, n):
	best_seed = 0
	best_diff = 100
	for i in range(1000):
		np.random.seed(i)
		folds = partition(list_in, n)
		lies  = []
		truth = []
		for p in folds:
			lies.append(sum(x[0] for x in p))
			truth.append(sum(x[1] for x in p))
		diff = sum(abs(lies[i]-truth[i]) for i in range(n)) 
		if diff < best_diff:
			best_seed = i
			best_diff = diff
			#print(i, lies, truth)
	return i



if __name__ == '__main__':
	
	df = pd.read_csv('../data/annotations_fixed.csv')

	### split dataset into cross-validation folds
	listA = [0,1,2,3,4,7,9,13,14,16,17,18,19,20,22,23,24,25,26,27,28,29]
	listB = list(range(0,35))

	dictA = {i:[0,0,i] for i in listA}
	dictB = {i:[0,0,i] for i in listB}

	for i in range(df.shape[0]):
		dictB[df['usernum'][i]][df['truth'][i]] += 1
		if df['usernum'][i] in listA:
			dictA[df['usernum'][i]][df['truth'][i]] += 1

	setA = [x for x in dictA.values()]
	setB = [x for x in dictB.values()]
	
	seedA = search_random_seed(setA, 2)
	np.random.seed(seedA)
	foldsA = partition(setA, 2)
	
	seedB = search_random_seed(setB, 3)
	np.random.seed(seedA)
	foldsB = partition(setB, 3)
	
	# for i in range(len(foldsA)):
	# 	print("SplitA : {} --- {}".format(i, foldsA[i]))
	for i in range(len(foldsB)):
		print("SplitB : {} --- {}".format(i, foldsB[i]))
	