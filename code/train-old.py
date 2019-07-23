import torch

import model.data_loader as data_loader


if __name__ == "__main__":

	data_path 	   = '/dresden/users/as2947/MURI/code/data'
	players_to_use = 'train.txt'
	labels_file    = 'labels.txt'

	train_dataset = data_loader.MURI_Dataset(data_path, players_to_use, labels_file)

	#test_dataset = data.MURI_Dataset(data_path, players_to_use, labels_file, 32)
	# train_data, test_data = get_train_test_data(test_subject_id, cfg)
	# train_dataset = DHS_Dataset(train_data, use_data_aug = True, time_len = 8, sample_strategy = "equi_T")
	# test_dataset = DHS_Dataset(test_data, use_data_aug = False, time_len = 8, sample_strategy = "equi_T")
	print("# of training data: {}".format(len(train_dataset)))

	for i in range(len(train_dataset)):
		sample = train_dataset[i]
		print("YOLO")
		print(i, sample['landmarks'].shape, sample['label'].shape)
		
		

	#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True, pin_memory=False)
	#val_loader = torch.utils.data.DataLoader( test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)