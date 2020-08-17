
def get_splits(splits_file):
    """
    Return:
        splits: dictionary with keys->split_num and values->list with corresponding videos
    """
    splits_num = int(splits_file.split("_")[1].split(".")[0])
    with open(splits_file, 'r') as spl:
        lines = spl.readlines()
    splits = {i : [] for i in range(splits_num)}
    for l in lines:
        splits[int(l.split()[1])].append(l.split()[0])
    # pdb.set_trace()
    return splits





if __name__ == '__main__':

    # set the random seed for reproducible experiments
    torch.manual_seed(777)
    if opt.use_gpu:
        torch.cuda.manual_seed(777)
	
	splits_dict = utils.get_splits(opt.splits)
	folds_num = len(splits_dict.keys())
    for i in range(folds_num):
        x_val = splits_dict[i].copy()
        x_train = []

        for j in range(folds_num):
            if j!=i:
                x_train = x_train + splits_dict[j]


        # re-initialize model for every fold here

       	# Train and evaluate model