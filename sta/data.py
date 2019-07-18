import os
import torch.utils.data as data
import pandas as pd
import numpy as np
import pdb

class MURI_Dataset(data.Dataset):
    def __init__(self, dataroot, filelist, labellist, time_len=8, time_step=6):
        super(MURI_Dataset, self).__init__()
        self.dataroot = dataroot
        with open(filelist, 'r') as f:
            self.filelist = [l.rstrip('\n') for l in f.readlines()]
        self.time_len = time_len
        self.time_step = time_step
        self.cols = [f'X_{i}' for i in range(68)] + [f'Y_{i}' for i in range(68)] + [f'Z_{i}' for i in range(68)]
        with open(labellist, 'r') as f:
            labels = f.readlines()
        self.labels = {l.split()[0]: int(l.split()[1]) for l in labels}
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        name, time_start = self.filelist[index].split()
        label = self.labels[name.replace('_R3', '')]
        filename = os.path.join(self.dataroot, name, f'{name}.csv')
        with open(filename, 'r') as f:
            h = [hh.strip(' ') for hh in f.readline().rstrip('\n').split(',')]
        col_idx = [h.index(hh) for hh in self.cols]
        csv = np.genfromtxt(filename, delimiter=',')[1:,col_idx]
        frame_list = []
        for i in range(int(time_start), int(time_start)+self.time_len*self.time_step, self.time_step):
            one_frame = csv[i]
            frame_list.append(one_frame.reshape((3, 68)).T)
            
        frame_list_ = np.concatenate(frame_list)
        
        return {'skeleton': frame_list_, 'label': label}
