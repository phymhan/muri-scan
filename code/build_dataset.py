
import os
import random
import shutil

CHUNK        = 900
LABELS       = 'labels.txt'
FILES2USE    = 'ok_files.txt'
FILES_FRAMES = 'filelist_with_frames.txt'


def balance_dataset(name, filenames, chunks, labels):
    with open('./data/' + name + '.txt', 'w') as f:
        for file in filenames:
            for i in range(chunks[file]):
                f.write(file + ' ' + str(i*CHUNK) + '\n')

    with open('./data/' + name + '.txt', 'r') as f:
        instances = [l.rstrip('\n') for l in f.readlines()]
    
    sp = []
    vill = []
    for inst in instances:
        if labels[inst.split()[0].replace('_R3', '')] == 1:
            sp.append(inst)
        else:
            vill.append(inst)
    keep = min(len(sp), len(vill))
    
    sp.sort()
    vill.sort()
    random.seed(775)
    random.shuffle(sp)
    random.shuffle(vill)
    
    # balalnce dataset
    sp = sp[:keep]
    vill = vill[:keep]
    keep_files = sp + vill
    keep_files.sort()
    random.shuffle(keep_files)    
    
    with open('./data/' + name + '.txt', 'w') as f:
        for file in keep_files:
            f.write(file+'\n')



if __name__ == '__main__':

    with open('./data/metadata/' + FILES2USE, 'r') as f:
        filelist = [l.rstrip('\n').split()[0] for l in f.readlines()]

    with open('./data/metadata/' +   FILES_FRAMES, 'r') as f:
        chunks = f.readlines()
    chunks = {c.split()[0]: int( int(c.split()[1]) / 900 ) for c in chunks}

    with open('./data/' + LABELS, 'r') as f:
        labels = f.readlines()
    labels = {l.split()[0]: int(l.split()[1]) for l in labels}

    spies = []
    villagers = []
    for player in filelist:
        if labels[player.replace('_R3', '')] == 1:
            spies.append(player)
        else:
            villagers.append(player)

    print("# of spies: {}\n# of villagers: {}".format(len(spies), len(villagers)))
    assert len(spies) + len(villagers) == 282, "The # of spies and villagers is wrong!"

    keep_num = min(len(spies), len(villagers))
    print("Will only keep {} villagers and {} spies".format(keep_num, keep_num))

    spies.sort()
    villagers.sort()
    random.seed(775)
    random.shuffle(spies)
    random.shuffle(villagers)
    # balance dataset
    villagers = villagers[:keep_num]

    split_1 = int(0.74 * keep_num)
    split_2 = int(0.87 * keep_num)
    train_filenames = spies[:split_1] + villagers[:split_1]
    val_filenames   = spies[split_1:split_2] + villagers[split_1:split_2]
    test_filenames  = spies[split_2:] + villagers[split_2:]
    assert len(train_filenames) + len(val_filenames) + len(test_filenames) == 2*keep_num, "Wrong dataset split!"

    balance_dataset('train', train_filenames, chunks, labels)
    balance_dataset('val', val_filenames, chunks, labels)
    balance_dataset('test', test_filenames, chunks, labels)
    
    '''
    path = os.getcwd()
    train_dir = os.path.join(path, 'data', 'train')
    val_dir   = os.path.join(path, 'data', 'val')
    test_dir  = os.path.join(path, 'data', 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    csv_dir = os.path.join(path,'data/openface/')
    for file in train_filenames:
        shutil.copy(csv_dir + file +'.csv', train_dir)
    for file in val_filenames:
        shutil.copy(csv_dir + file +'.csv', val_dir)
    for file in test_filenames:
        shutil.copy(csv_dir + file +'.csv', test_dir)
    '''
