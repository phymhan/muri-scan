import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pdb

class MURI_Dataset(Dataset):
    
    def __init__(self, dataroot, filelist, labellist, time_len=24, time_step=5):
        
        self.dataroot = dataroot
        self.time_len = time_len
        self.time_step = time_step
        
        with open(os.path.join(dataroot, filelist), 'r') as f:
            self.filelist = [l.rstrip('\n') for l in f.readlines()]
        
        with open(os.path.join(dataroot, labellist), 'r') as f:
            labels = f.readlines()
        self.labels = {l.split()[0]: int(l.split()[1]) for l in labels}


    def __len__(self):
        return len(self.filelist)


    def __getitem__(self, idx):

        player, time_start = self.filelist[idx].split()
        label = self.labels[player.replace('_R3', '')]
        filename = os.path.join(self.dataroot, 'npz', f'{player}-{time_start}.npz')
        data = np.load(filename)
        
        chunk_size = data['landmarks'].shape[0]
        start_idx  = np.random.randint(0, chunk_size - self.time_len*self.time_step)

        return np.array(data['landmarks'][start_idx:start_idx+self.time_len*self.time_step:self.time_step], dtype='float32'), np.array(data['features'][start_idx:start_idx+self.time_len*self.time_step:self.time_step, ...], dtype='float32'), label



def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            if split == 'train':
                dl = DataLoader(MURI_Dataset(data_dir, 'train.txt', 'labels.txt', params.time_len, params.time_step) ,
                                batch_size=params.batch_size, num_workers=params.num_workers, 
                                shuffle=True, pin_memory=params.cuda
                               )
            else:
                dl = DataLoader(MURI_Dataset(data_dir, split+'.txt', 'labels.txt', params.time_len, params.time_step) , 
                                batch_size=params.batch_size, num_workers=params.num_workers, 
                                shuffle=False,  pin_memory=params.cuda
                                )
            dataloaders[split] = dl

    return dataloaders
