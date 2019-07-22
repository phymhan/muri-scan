import os

src = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/openface/clips-r3'
names = os.listdir(src)

for name in names:
    img_path = os.path.join(src, name, f'{name}_aligned')
    vid_path = os.path.join(src, name, f'aligned.mp4')
    os.system(f'ffmpeg -r 30 -f image2 -i {img_path}/frame_det_00_%06d.bmp -vcodec libx264 -crf 25 -pix_fmt yuv420p {vid_path}')