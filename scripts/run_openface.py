import os

src = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/clips/r3'
dst = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/openface/clips-r3_wrong-fps'  # for wrong fps only
with open('/media/ligong/Picasso/Active/muri-scan/sourcefiles/wrong-fps.txt', 'r') as f:
    filenames = [l.rstrip('\n') for l in f.readlines()]

for filename in filenames:
    in_file = os.path.join(src, filename)
    name_ = filename.split('_')
    new_filename = f'{name_[0]}-{int(name_[1]):03d}_{name_[2]}_{name_[3]}'
    out_file = os.path.join(dst, new_filename)
    os.system(f'/media/ligong/Picasso/Active/OpenFace/build/bin/FeatureExtraction -f "{in_file}" -out_dir "{out_file}" -verbose')
    print(filename)