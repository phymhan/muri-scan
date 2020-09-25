import os
import numpy as np
from torch.utils.data import Dataset
<<<<<<< HEAD
import pdb
import cv2
import utils
from localbinarypatterns import LocalBinaryPatterns
=======
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24


class BagOfLies(Dataset):
    """
    We use only 100 frames from each video
    """
    def __init__(self, dataroot, file_list, label_list, num_frames=100, time_stride=1):
        
        self.dataroot = dataroot
        self.num_frames = num_frames
        self.time_stride = time_stride
        self.filelist = list(map(lambda x: '-'.join(x.split('/')), file_list))

        with open(label_list, 'r') as f:
            labels = f.readlines()
        self.labels = {'-'.join((l.split()[0]).split('/')): int(l.split()[1]) for l in labels}
        
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
        
        gaze = np.array(data['arr_0'][start_idx:start_idx + self.num_frames * self.time_stride:self.time_stride, 7:9], dtype='float32')
        aus  = np.array(data['arr_0'][start_idx:start_idx + self.num_frames * self.time_stride:self.time_stride, 15:], dtype='float32')
        features = np.concatenate((gaze, aus), axis=1)
        
        return features.T, label
<<<<<<< HEAD


class BagOfLiesImage(Dataset):
    """
    We use only 100 frames from each video
    """
    def __init__(self, dataroot, file_list, label_list, num_frames=100, time_stride=1):
        
        self.dataroot = dataroot
        self.num_frames = num_frames
        self.time_stride = time_stride
        self.filelist = list(map(lambda x: '-'.join(x.split('/')), file_list))

        with open(label_list, 'r') as f:
            labels = f.readlines()
        self.labels = {'-'.join((l.split()[0]).split('/')): int(l.split()[1]) for l in labels}
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        """ 
        Return:
            features : numpy array of size (feature_dim, num_frames)
            label    : 0->deceptive, 1->truth
        """
        desc = LocalBinaryPatterns(24, 8)

        name  = self.filelist[idx]
        label = self.labels[name]
        name = name.replace('-', '/')
        filename = os.path.join(self.dataroot, name+'.avi')

        numframe = len(os.listdir(os.path.join(self.dataroot, name, 'video_aligned')))
        chunksize = int(numframe / 20.)
        idxs = range(1, numframe+1, chunksize)[:20]
        hists = []

        for idx in idxs:
            idx = idx + np.random.randint(0, chunksize-1)
            idx = min(idx, numframe)
            filename = os.path.join(self.dataroot, name, 'video_aligned', f'frame_det_00_{idx:06d}.bmp')
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hists.append(desc.describe(gray))
        
        hist = np.concatenate(hists)
        return hist, label
=======
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24
