import os
import numpy as np
from torch.utils.data import Dataset


class RLT(Dataset):
    """
    We use only 180 frames from each video
    """
    def __init__(self, dataroot, file_list, label_list, num_frames=180, time_stride=1):
        
        self.dataroot = dataroot
        self.num_frames = num_frames
        self.time_stride = time_stride
        self.filelist = file_list

        with open(label_list, 'r') as f:
            labels = f.readlines()
        self.labels = {l.split()[0]: int(l.split()[1]) for l in labels}
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        """ 
        Return:
            features : numpy array of size (feature_dim, num_frames)
            label    : 0->deceptive, 1->truth
        """
        name  = self.filelist[idx]
        label = self.labels[name]

        filename = os.path.join(self.dataroot, 'openface/npz-use/', name+'.npz')
        data = np.load(filename)
        
        video_frames = data['arr_0'].shape[0]
        if video_frames <= self.num_frames:
            start_idx = 0     
        else:
            start_idx = np.random.randint(0, video_frames - self.num_frames * self.time_stride)
        # features  = np.array(data['arr_0'][start_idx:start_idx + self.num_frames * self.time_stride:self.time_stride, 15:], dtype='float32')
        gaze = np.array(data['arr_0'][start_idx:start_idx + self.num_frames * self.time_stride:self.time_stride, 7:9], dtype='float32')
        aus  = np.array(data['arr_0'][start_idx:start_idx + self.num_frames * self.time_stride:self.time_stride, 15:], dtype='float32')
        features = np.concatenate((gaze, aus), axis=1)
        
        return features.T, label
