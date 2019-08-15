from torch.utils.data import Dataset
import os
import numpy as np


class ClipDataset(Dataset):
    def __init__(self, dataroot, sourcefile, labelfile, time_len=24, time_step=5):
        self.dataroot = dataroot
        self.time_len = time_len
        self.time_step = time_step

        with open(sourcefile, 'r') as f:
            self.filelist = [l.rstrip('\n') for l in f.readlines()]

        with open(labelfile, 'r') as f:
            labels = f.readlines()
        self.labels = {l.split()[0]: int(l.split()[1]) for l in labels}

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        # return global_feat of size (time_len, global_feature_dim)
        player, time_start = self.filelist[idx].split()
        label = self.labels[player.replace('_R3', '')]
        filename = os.path.join(self.dataroot, f'{player}-{time_start}.npz')
        data = np.load(filename)
        chunk_size = data['features'].shape[0]
        start_idx = np.random.randint(0, chunk_size - self.time_len * self.time_step)
        global_feat = np.array(
            data['features'][start_idx:start_idx + self.time_len * self.time_step:self.time_step, ...], dtype='float32')

        return global_feat, label


class VideoDataset(Dataset):
    def __init__(self, dataroot, sourcefile, labelfile, time_len=24, time_step=5):
        self.dataroot = dataroot
        self.time_len = time_len
        self.time_step = time_step

        with open(labelfile, 'r') as f:
            labels = f.readlines()
        self.labels = {l.split()[0]: int(l.split()[1]) for l in labels}

        with open(sourcefile, 'r') as f:
            self.filelist = [l.rstrip('\n') for l in f.readlines()]

        videolist = []
        mapping = []  # mapping index to video name
        clip_num = []
        for clip in self.filelist:
            player, time_start = clip.split()
            if player in mapping:
                id = mapping.index(player)
                videolist[id].append(int(time_start))
            else:
                mapping.append(player)
                videolist.append([int(time_start)])
        cnt = [0, 0]
        for i, cliplist in enumerate(videolist):
            cliplist.sort()
            clip_num.append(len(cliplist))
            player = mapping[i]
            label = self.labels[player.replace('_R3', '')]
            cnt[label] += 1
        print(f'--> 0: {cnt[0]}, 1: {cnt[1]}')
        self.videolist = videolist
        self.mapping = mapping
        self.clip_num = clip_num
        self.min_clip_num = min(clip_num)

    def __len__(self):
        return len(self.videolist)

    def __getitem__(self, idx):
        # sample min_clip_num clips and return feature of size (min_clip_num, time_len, global_feature_dim)
        time_start_list = np.random.choice(self.videolist[idx], self.min_clip_num, replace=False)
        player = self.mapping[idx]
        label = self.labels[player.replace('_R3', '')]
        global_feat = []
        for time_start in time_start_list:
            filename = os.path.join(self.dataroot, f'{player}-{time_start}.npz')
            data = np.load(filename)
            chunk_size = data['features'].shape[0]
            start_idx = np.random.randint(0, chunk_size - self.time_len * self.time_step)
            global_feat.append(np.array(
                data['features'][start_idx:start_idx + self.time_len * self.time_step:self.time_step, ...],
                dtype='float32'))
        return np.stack(global_feat), label


class VideoDatasetV2(Dataset):
    def __init__(self, dataroot, sourcefile, clip_num=5, time_len=60, time_step=1):
        self.dataroot = dataroot
        self.clip_num = clip_num
        self.time_len = time_len
        self.time_step = time_step

        with open(sourcefile, 'r') as f:
            self.filelist = [l.rstrip('\n') for l in f.readlines()]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # sample min_clip_num clips and return feature of size (clip_num, time_len, global_feature_dim)
        player = self.filelist[index].split()[0]
        label = int(self.filelist[index].split()[1])
        feat = np.load(os.path.join(self.dataroot, f'{player}.npz'))['features']
        frame_num = feat.shape[0]
        frame_num_per_clip = int(np.ceil(frame_num / self.clip_num))
        feat_list = []
        for i in range(self.clip_num):
            chunk_size = min(frame_num, (i+1)*frame_num_per_clip) - i*frame_num_per_clip
            start_idx = np.random.randint(0, chunk_size - self.time_len * self.time_step) + i*frame_num_per_clip
            feat_list.append(np.array(
                feat[start_idx:start_idx + self.time_len * self.time_step:self.time_step, ...], dtype='float32'))
        return np.stack(feat_list), label
