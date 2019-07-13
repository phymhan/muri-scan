src_dir = '/media/ligong/Picasso/Share/MURI/clips-r3';
src_list = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/r3.txt';
dst_dir = '/media/ligong/Picasso/Share/cbimfs/Research/MURI/openface/clips-r3';

in_files = strsplit(fileread(src_list), '\n');
if isempty(in_files{end}), in_files = in_files(1:end-1); end

for in_file = in_files
    in_file_ = strsplit(in_file{1}, '_');
    in_file = sprintf('%s-%03d_%d_%s%s', in_file_{1}, str2num(in_file_{2}), str2num(in_file_{3}), in_file_{4}, in_file_{5});
    in_file = fullfile(src_dir, in_file);
    [~, name, ~] = fileparts(in_file);
    output_dir = fullfile(dst_dir, name);
    muri_1(in_file, output_dir);
end
