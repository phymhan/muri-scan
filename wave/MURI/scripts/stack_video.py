import cv2
import pdb
import numpy as np
import os

vid1src = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/all_clips'
vid2src = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/all_openface'
vid3src = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/attn_plt'
outroot2 = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/stack2'
outroot3 = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/stack3'

def stack_video3(p1, p2, p3, po):
    v1 = cv2.VideoCapture(p1)
    v2 = cv2.VideoCapture(p2)
    v3 = cv2.VideoCapture(p3)
    vo = None

    while 1:
        _, i1 = v1.read()
        _, i2 = v2.read()
        success, i3 = v3.read()
        if not success:
            break
        w = int(1.0*i1.shape[1]*i3.shape[0]/i1.shape[0])
        h = i3.shape[0]
        i1_ = cv2.resize(i1, (w, h))
        i2_ = cv2.resize(i2, (w, h))
        # pdb.set_trace()
        o = np.concatenate((i1_, i2_, i3), axis=1)
        if vo is None:
            vo = cv2.VideoWriter(po, cv2.VideoWriter_fourcc(*'DIVX'), 24, (o.shape[1], o.shape[0]))
        vo.write(o)
    v1.release()
    v2.release()
    v3.release()
    if vo is not None:
        vo.release()

def stack_video2(p2, p3, po):
    v2 = cv2.VideoCapture(p2)
    v3 = cv2.VideoCapture(p3)
    vo = None

    while 1:
        _, i2 = v2.read()
        success, i3 = v3.read()
        if not success:
            break
        w = int(1.0*i2.shape[1]*i3.shape[0]/i2.shape[0])
        h = i3.shape[0]
        i2_ = cv2.resize(i2, (w, h))
        # pdb.set_trace()
        o = np.concatenate((i2_, i3), axis=1)
        if vo is None:
            vo = cv2.VideoWriter(po, cv2.VideoWriter_fourcc(*'DIVX'), 24, (o.shape[1], o.shape[0]))
        vo.write(o)
    v2.release()
    v3.release()
    if vo is not None:
        vo.release()


file_list = os.listdir(vid3src)
name_list = [n.replace('.avi', '') for n in file_list]
for filename in name_list:
    print(filename)
    stack_video3(os.path.join(vid1src, filename+'.mp4'), os.path.join(vid2src, filename, filename+'.avi'), os.path.join(vid3src, filename+'.avi'), os.path.join(outroot3, filename+'.avi'))
    stack_video2(os.path.join(vid2src, filename, filename+'.avi'), os.path.join(vid3src, filename+'.avi'), os.path.join(outroot2, filename+'.avi'))
