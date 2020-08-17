vidroot1 = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/all_clips';
vidroot2 = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/all_openface';
outroot = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/attn_vis';
pltroot = '/media/ligong/Picasso/Datasets/Real-life_Deception_Detection_2016/attn_plt';

file_list = dir(fullfile(vidroot1, '*.mp4'));
fl = {};
for i = 1:length(file_list)
    filename = file_list(i).name;
    fl{end+1} = strrep(filename, '.mp4', '');
end

leg = {'01 Inner Brow Raiser', ...
    '02 Outer Brow Raiser', ...
    '04 Brow Lowerer', ...
    '05 Upper Lid Raiser', ...
    '06 Cheek Raiser', ...
    '07 Lid Tightener', ...
    '09 Nose Wrinkler', ...
    '10 Upper Lip Raiser', ...
    '12 Lip Corner Puller', ...
    '14 Dimpler', ...
    '15 Lip Corner Depressor', ...
    '17 Chin Raiser', ...
    '20 Lip stretcher', ...
    '23 Lip Tightener', ...
    '25 Lips part', ...
    '26 Jaw Drop', ...
    '45 Blink', ...
    'Attention'};
leg1 = {'01 Inner Brow Raiser', ...
    '02 Outer Brow Raiser', ...
    '04 Brow Lowerer', ...
    '05 Upper Lid Raiser', ...
    '06 Cheek Raiser', ...
    '07 Lid Tightener', ...
    '09 Nose Wrinkler', ...
    '10 Upper Lip Raiser', ...
    '12 Lip Corner Puller', ...
    'Attention'};
leg2 = {'14 Dimpler', ...
    '15 Lip Corner Depressor', ...
    '17 Chin Raiser', ...
    '20 Lip stretcher', ...
    '23 Lip Tightener', ...
    '25 Lips part', ...
    '26 Jaw Drop', ...
    '45 Blink', ...
    'Attention'};
w = gausswin(10);
colors = {[0 0.4470 0.7410], ...
    [0.8500 0.3250 0.0980], ...
    [0.9290 0.6940 0.1250], ...
    [0.4940 0.1840 0.5560], ...
    [0.4660 0.6740 0.1880], ...
    [0.3010 0.7450 0.9330], ...
    [0.6350 0.0780 0.1840], ...
    [1 0 0], ...
    [0 1 0]};
tw = 90;
for j = 45:length(fl)
    filename = fl{j};
    fprintf([filename '\n']);
    vobj = VideoWriter(fullfile(pltroot, filename));
    vobj.FrameRate = 24;
    open(vobj);
    
    if exist(fullfile(outroot, [filename '_au.npy']), 'file')
        y_au = readNPY(fullfile(outroot, [filename '_au.npy']));
        y_att = readNPY(fullfile(outroot, [filename '_att.npy']));
    else
        continue
    end
    
    num_frame = min(length(y_att), size(y_au, 2));
    y_att = y_att(1:num_frame);
    y_au = y_au(:,1:num_frame);
    
    y_au = y_au';
    y_au = y_au / 5;
    y_au = filter(w, 1, y_au);
    y_au = min(1, y_au);
    
    % m = (y_att-min(y_att))/(max(y_att)-min(y_att));
    m_sort = sort(y_att);
    m_min = m_sort(max(1,floor(num_frame*0.2)));
    m_max = m_sort(floor(num_frame*0.8));
    m = min(max(m_min, y_att), m_max);
    m = (m-m_min)/(m_max-m_min);
    
    set(gcf, 'position', [-1800         50        1600         800])
    
    subplot(2,1,1)
    % xlim([0 num_frame])
    xlim([0 tw])
    xl = get(gca, 'xtick');
    set(gca, 'xtick', xl(1:end-1))
    ylim([0 1])
    box on
    grid on
    k = 1;
    line(repmat(t, 1, 9), y_au(1:k, 1:9), 'linewidth', 2)
    line(t, m(1:k, :), 'linewidth', 4, 'linestyle', '-', 'color', [0.7 0.7 0.7])
    
    subplot(2,1,2)
    % xlim([0 num_frame])
    xlim([0 tw])
    ylim([0 1])
    box on
    grid on
    k = 1;
    line(repmat(t, 1, 8), y_au(1:k, 10:17), 'linewidth', 2)
    line(t, m(1:k, :), 'linewidth', 4, 'linestyle', '-', 'color', [0.7 0.7 0.7])
    
    for k = 1:num_frame
        % cla
        t = (1:k)';
        % line(repmat(t, 1, 17), y_au(1:k, :), 'linewidth', 2)
        % line(t, m(1:k, :), 'linewidth', 4, 'linestyle', '-', 'color', [0.7 0.7 0.7])
        subplot(2,1,1)
        cla
        xlim([max(0, k-tw) max(tw, k)])
        xl = [max(0, k-tw) max(tw, k)];
        set(gca, 'xtick', xl(1):5:xl(2))
        for l = 1:9
%             if k < tw
%                 t_ = t;
%                 y_ = y_au(1:k,l+0);
%             else
%                 t_ = t(end-tw+1:end);
%                 y_ = y_au(end-tw+1:end,l+0);
%             end
%             line(t_, y_, 'linewidth', 2, 'color', colors{l})
            line(t, y_au(1:k, l+0), 'linewidth', 2, 'color', colors{l})
        end
%         if k < tw
%             m_ = m(1:k);
%         else
%             m_ = m(end-tw+1:end);
%         end
%         line(t_, m_, 'linewidth', 4, 'color', [1 1 1]*0.7);
        line(t, m(1:k), 'linewidth', 4, 'color', [1 1 1]*0.7);
        legend(leg1, 'location', 'eastoutside')
        
        subplot(2,1,2)
        cla
        xlim([max(0, k-tw) max(tw, k)])
        % xtik = get(gca, 'xtick');
        xl = [max(0, k-tw) max(tw, k)];
        set(gca, 'xtick', xl(1):5:xl(2))
        for l = 1:8
%             if k < tw
%                 t_ = t;
%                 y_ = y_au(1:k,l+9);
%             else
%                 t_ = t(end-tw+1:end);
%                 y_ = y_au(end-tw+1:end,l+9);
%             end
%             line(t_, y_, 'linewidth', 2, 'color', colors{l})
            line(t, y_au(1:k, l+9), 'linewidth', 2, 'color', colors{l})
        end
%         if k < tw
%             m_ = m(1:k);
%         else
%             m_ = m(end-tw+1:end);
%         end
%         line(t_, m_, 'linewidth', 4, 'color', [1 1 1]*0.7);
        line(t, m(1:k), 'linewidth', 4, 'color', [1 1 1]*0.7);
        legend(leg2, 'location', 'eastoutside')
        
        frame = getframe(gcf);
        writeVideo(vobj, frame);
    end
    
    close(vobj);
    
end
