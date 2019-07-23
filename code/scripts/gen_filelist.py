
import os

CHUNK        = 900
FILES2USE    = 'ok_files.txt'
FILES_FRAMES = 'filelist_with_frames.txt'

if __name__ == '__main__':

    with open('../data/metadata/' + FILES2USE, 'r') as f:
        filelist = [l.rstrip('\n').split()[0] for l in f.readlines()]

    with open('../data/metadata/' +   FILES_FRAMES, 'r') as f:
        chunks = f.readlines()
    chunks = {c.split()[0]: int( int(c.split()[1]) / 900 ) for c in chunks}

    with open('../data/filelist.txt', 'w') as f:
        for file in filelist:
            for i in range(chunks[file]):
                f.write(file + ' ' + str(i*CHUNK) + '\n')
