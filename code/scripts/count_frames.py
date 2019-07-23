
with open('../data/metadata/filelist_with_frames.txt', 'r') as fl:
	frames = fl.readlines()
frames = {l.split()[0]: int(l.split()[1]) for l in frames}

with open('../data/train.txt', 'r') as f1, open('../data/val.txt', 'r') as f2, open('../data/test.txt', 'r') as f3:
	train_list = list(map(lambda x: x.rstrip().split()[0], f1.readlines()))
	val_list   = list(map(lambda x: x.rstrip().split()[0], f2.readlines()))
	test_list  = list(map(lambda x: x.rstrip().split()[0], f3.readlines()))

fr_train = [frames[t] for t in train_list]
fr_val   = [frames[t] for t in val_list]
fr_test  = [frames[t] for t in test_list]

print("Train : min {} -- max {}\nVal : min {} -- max {}\nTest : min {} -- max {}".format(
						min(fr_train), max(fr_train), min(fr_val), max(fr_val), min(fr_test), max(fr_test)))