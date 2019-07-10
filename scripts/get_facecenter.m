% bbox: [upper_left_x, upper_left_y, bottum_right_x, bottum_right_y]

src = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/clips/r3';
dst = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/bbox';
if ~exist(dst, 'dir')
    mkdir(dst)
end
vids = dir(fullfile(src, '*.mp4'))';
N = length(vids);
n = 0;

for vid_ = vids
    txtname = fullfile(dst, strrep(vid_.name, '.mp4', '.txt'));
    if exist(txtname, 'file')
        n = n + 1;
        continue
    end
    if vid_.bytes / 1e6 < 10
        continue
    end
    n = n + 1;
    v = VideoReader(fullfile(src, vid_.name));
    fr = readFrame(v);
    imshow(fr);
    fprintf('--> [%.2f%%] %s\n', n/N*100, vid_.name);
    %     waitforbuttonpress
    [x, y] = ginput(2);
    fid = fopen(txtname, 'w');
    fprintf(fid, '%.2f %.2f %.2f %.2f', x(1), y(1), x(2), y(2));
    fclose(fid);
end
