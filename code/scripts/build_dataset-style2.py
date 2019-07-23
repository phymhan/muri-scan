
import os
import random

LABELS       = 'labels.txt'
FILES2USE    = 'filelist.txt'


if __name__ == '__main__':

    with open('./data/' + FILES2USE, 'r') as f:
        filelist = [l.rstrip('\n') for l in f.readlines()]

    with open('./data/' + LABELS, 'r') as f:
        labels = f.readlines()
    labels = {l.split()[0]: int(l.split()[1]) for l in labels}

    spies = []
    villagers = []
    for player in filelist:
        if labels[player.split()[0].replace('_R3', '')] == 1:
            spies.append(player)
        else:
            villagers.append(player)

    print("# of spies: {}\n# of villagers: {}".format(len(spies), len(villagers)))
    
    keep_num = min(len(spies), len(villagers))
    print("Will only keep {} villagers and {} spies".format(keep_num, keep_num))

    spies.sort()
    villagers.sort()
    random.seed(775)
    random.shuffle(spies)
    random.shuffle(villagers)
    # balalnce dataset
    villagers = villagers[:keep_num]

    split_1 = 1900
    split_2 = split_1 + 162 
    train_filenames = spies[:split_1] + villagers[:split_1]
    val_filenames   = spies[split_1:split_2] + villagers[split_1:split_2]
    test_filenames  = spies[split_2:] + villagers[split_2:]
    
    with open('./data/train.txt', 'w') as f:
        for file in train_filenames:
            f.write(file+'\n')

    with open('./data/val.txt', 'w') as f:
        for file in val_filenames:
            f.write(file+'\n')

    with open('./data/test.txt', 'w') as f:
        for file in test_filenames:
            f.write(file+'\n')

    
    # -------------------------------
    # check if generated split is ok

    with open('./data/train.txt', 'r') as f:
        lines_train = [l.rstrip('\n').split()[0] for l in f.readlines()]
    
    with open('./data/val.txt', 'r') as f:
        lines_val = [l.rstrip('\n').split()[0] for l in f.readlines()]

    with open('./data/test.txt', 'r') as f:
        lines_test = [l.rstrip('\n').split()[0] for l in f.readlines()]

    sp = []
    vill = []
    for player in lines_train:
        if labels[player.replace('_R3', '')] == 1:
            sp.append(player)
        else:
            vill.append(player)
    print("Training Set --- # of spies: {}\n# of villagers: {}".format(len(sp), len(vill)))

    sp = []
    vill = []
    for player in lines_val:
        if labels[player.replace('_R3', '')] == 1:
            sp.append(player)
        else:
            vill.append(player)
    print("Validation Set --- # of spies: {}\n# of villagers: {}".format(len(sp), len(vill)))

    sp = []
    vill = []
    for player in lines_test:
        if labels[player.replace('_R3', '')] == 1:
            sp.append(player)
        else:
            vill.append(player)
    print("Test Set --- # of spies: {}\n# of villagers: {}".format(len(sp), len(vill)))
    