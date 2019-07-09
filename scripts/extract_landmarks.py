import os
import sys
import glob
from datetime import datetime
import skvideo.io
import shutil
from facial_landmarks import get_heatmap_from_image
import face_alignment
from PIL import Image
import torch
import numpy
import pdb


def upsample_image(tensor, size):
    tensor = torch.nn.functional.interpolate(input=tensor.unsqueeze_(0), size=size, mode='bilinear', align_corners=True)
    return tensor[0, ...]


src = '/dresden/users/lh599/Research/MURI/clips/r3'
dst_vid = '/dresden/users/lh599/Research/MURI/landmarks'
dst_txt = '/dresden/users/lh599/Research/MURI/landmarks_txt'


if not os.path.exists(dst_vid):
    os.makedirs(dst_vid)
if not os.path.exists(dst_txt):
    os.makedirs(dst_txt)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, face_detector='sfd', flip_input=True, device='cuda')

vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(src, '*.mp4'))]
for filename in vidfiles:
    file_in = os.path.join(src, filename)
    file_out = os.path.join(dst_vid, filename)
    txt_out = os.path.join(dst_txt, filename.replace('.mp4', '.txt'))
    video_data = skvideo.io.vread(file_in)
    writer = skvideo.io.FFmpegWriter(file_out)
    with open(txt_out, 'w') as f:
        for i, image in enumerate(video_data):
            lm, pt = get_heatmap_from_image(image, fa, ret_pts=True)
            if lm is None:
                lm = torch.zeros([1, image.shape[0], image.shape[1]])
                pt = torch.zeros([68, 2])
            else:
                lm = torch.clamp(upsample_image(lm[0], (image.shape[0], image.shape[1])), 0, 1)
                pt = pt[0]
            frame = (lm.cpu().numpy() * 255).transpose((1, 2, 0)).astype(numpy.uint8)
            writer.writeFrame(frame)
            f.write(f'{i} ' + ' '.join(map(str, pt.view(-1).numpy())) + '\n')
            if i % 1000 == 0:
                print(f'--> frame {i}')
    writer.close()
    print('-> %s' % file_out)
