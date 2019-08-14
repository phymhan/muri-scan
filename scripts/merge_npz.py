import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../sourcefiles/train.txt')
parser.add_argument('--output', type=str, default='../sourcefiles/train_video.txt')
parser.add_argument('--input_dir', type=str, default='/dresden/users/lh599/Data/muri/npz-r3')
parser.add_argument('--output_dir', type=str, default='/dresden/users/lh599/Data/muri/npz-r3-video')
parser.add_argument('--labels', type=str, default='../sourcefiles/labels.txt')
args = parser.parse_args()

with open(args.labels, 'r') as f:
    labels = f.readlines()
labels = {l.split()[0]: int(l.split()[1]) for l in labels}

with open(args.input, 'r') as f:
    filelist = [l.rstrip('\n') for l in f.readlines()]

videolist = []
mapping = []  # mapping index to video name
clip_num = []
for clip in filelist:
    player, time_start = clip.split()
    if player in mapping:
        id = mapping.index(player)
        videolist[id].append(int(time_start))
    else:
        mapping.append(player)
        videolist.append([int(time_start)])

with open(args.output, 'w') as f:
    for i, cliplist in enumerate(videolist):
        cliplist.sort()
        clip_num.append(len(cliplist))
        player = mapping[i]
        label = labels[player.replace('_R3', '')]
        npz_list = []
        for time_start in cliplist:
            npz_list.append(np.load(os.path.join(args.input_dir, f'{player}-{time_start}.npz'))['features'])
        np.savez(os.path.join(args.output_dir, player) + '.npz', features=np.concatenate(npz_list))
        f.write(f'{player} {label}\n')
        print(f'--> {player}')
