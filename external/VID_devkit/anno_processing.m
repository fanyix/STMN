
data_root = '/home/SSD2/fanyi-data/ILSVRC2015/Data/VID/test';
anno_root = '/home/SSD2/fanyi-data/ILSVRC2015/Annotations/VID/test';
category_meta_data = '/home/fanyix/code/VID/data/VID_devkit_data/meta_vid.mat';

% read meta data
meta = load(category_meta_data);
meta = meta.synsets;
WNID = arrayfun(@(x)x.WNID, meta, 'uniformoutput',false);
category_name = arrayfun(@(x)x.name, meta, 'uniformoutput',false);
annotation = {};
finished_count = 0;
anno_mat_filename = '/home/SSD2/fanyi-data/ILSVRC2015/Annotations/VID/test/full.mat';


%%%%%%%%%%%%%%%%%%%%%%%%%% For train set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for kk = 1:4
%     subset_name = sprintf('ILSVRC2015_VID_train_%.4d', kk-1);
%     sub_data_root = fullfile(data_root, subset_name);
%     sub_anno_root = fullfile(anno_root, subset_name);
%     % get sub-directories
%     dir_list = dir(sub_data_root);
%     dir_list = dir_list(arrayfun(@(x) x.isdir && ~strcmp(x.name, '.') && ~strcmp(x.name, '..'), dir_list));
%     for i = 1:length(dir_list)
%         video_name = dir_list(i).name;
%         data_dir = fullfile(sub_data_root, video_name);
%         img_list = dir(fullfile(data_dir, '*.JPEG'));
%         
%         % sort frames
%         img_list = arrayfun(@(x)x.name, img_list, 'uniformoutput', false);
%         img_list_id = zeros(length(img_list), 1);
%         for j = 1:length(img_list)
%             [ignore, num] = fileparts(img_list{j});
%             num = str2double(num);
%             img_list_id(j) = num;
%         end
%         [sortval sortidx] = sort(img_list_id, 'ascend');
%         img_list = img_list(sortidx);
% 
%         % read annotations
%         anno_dir = fullfile(sub_anno_root, video_name);
%         rec = cell(length(img_list), 1);
%         img_size = zeros(length(img_list), 2);
%         for j = 1:length(img_list)
%             [ignore, filename] = fileparts(img_list{j});
%             anno_filename = fullfile(anno_dir, [filename '.xml']);
%             rec{j} = VOCreadxml(anno_filename);
%             img_size(j, :) = [str2double(rec{j}.annotation.size.height), str2double(rec{j}.annotation.size.width)];
%         end
% 
%         val_idx = find(cellfun(@(x)isfield(x.annotation,'object'), rec));
%         rec = rec(val_idx);
%         objs = cellfun(@(x)x.annotation.object, rec, 'uniformoutput', false);
%         objs = cat(2,objs{:});
%         obj_idx = unique(arrayfun(@(x)str2double(x.trackid), objs));
%         max_obj = length(obj_idx);
% 
%         objects = cell(max_obj, 1);
%         obj_category_names = cell(max_obj,1);
%         for j = 1:max_obj
%             objects{j}.boxes = [];
%             cur_obj_idx = obj_idx(j);
%             for k = 1:length(val_idx)
%                 trackid = arrayfun(@(x)str2double(x.trackid), rec{k}.annotation.object);
%                 obj_val_idx = find(cur_obj_idx == trackid);
%                 if ~isempty(obj_val_idx)
%                     if isempty(obj_category_names{j})
%                         obj_category_names{j} = rec{k}.annotation.object(obj_val_idx).name;
%                     end
%                     %if strcmp(rec{k}.annotation.object(obj_val_idx(1)).name,'n02958343')
%                     box = rec{k}.annotation.object(obj_val_idx).bndbox;
%                     width = str2double(rec{k}.annotation.size.width);
%                     height = str2double(rec{k}.annotation.size.height);
%                     size = [width height width height];
%                     scaled_box = [str2double(box.xmin), str2double(box.ymin), ...
%                             str2double(box.xmax), str2double(box.ymax)];
%                     scaled_box = bsxfun(@rdivide, scaled_box, size);
%                     objects{j}.boxes = [objects{j}.boxes; j, val_idx(k), scaled_box];
%                 end
%             end
%             
%             label = category_name{strcmp(obj_category_names{j}, WNID)};
%             objects{j}.obj_idx = j;
%             objects{j}.start_frame = objects{j}.boxes(1,2);
%             objects{j}.end_frame = objects{j}.boxes(end,2);
%             objects{j}.category = label;
%         end
%         
%         
%         annotation{finished_count+1}.obj = objects;
%         annotation{finished_count+1}.im_list = img_list;
%         annotation{finished_count+1}.video_name = sprintf('%s/%s', subset_name, video_name);
%         annotation{finished_count+1}.im_size = img_size;
%         finished_count = finished_count + 1;
%         
%         % % only keep sequences for particular categories
%         % objects = objects(strcmp(obj_category_names,'n02958343'));
%         % max_obj = length(objects);
% 
%         % % write 
%         % video_tag = dir_list(i).name;
%         % fprintf(fid, 'video_tag\n');
%         % fprintf(fid, '%s\n', video_tag);
%         % fprintf(fid, '%g %g\n', length(img_list),max_obj);
%         % for p = 1:length(img_list)
%         %     out_filename = fullfile(subset_name, video_name,img_list(p).name);
%         %     fprintf(fid, '%g %s\n', p, out_filename);
%         % end
%         % 
%         % % print annotation 
%         % for p = 1:max_obj
%         %     % determine the class label
%         %     label = find(strcmp(obj_category_names{p}, WNID));
%         % 
%         %     % let's do some analysis to figure out how many segments are there
%         %     boxes = objects{p}.boxes;
%         %     frame_id = boxes(:,2);
%         %     segs = frame_id(2:end) - frame_id(1:end-1);
%         %     endpoints = find(segs ~= 1);
%         %     endpoints = [0; endpoints; size(boxes,1)];
%         % 
%         % 
%         %     for pp = 1:length(endpoints)-1
%         %         fprintf(fid, 'object_tag\n');
%         %         obj_tag = sprintf('%g %g %g %g %g', p, endpoints(pp+1)-endpoints(pp), ...
%         %             frame_id(endpoints(pp)+1), frame_id(endpoints(pp+1)), label);
%         %         fprintf(fid, '%s\n', obj_tag);
%         % 
%         %         for q=(endpoints(pp)+1):endpoints(pp+1)
%         %             fprintf(fid, '%g %g %g %g %g %g\n',...
%         %                 objects{p}.boxes(q,1), objects{p}.boxes(q,2), objects{p}.boxes(q,3:4), ...
%         %                 objects{p}.boxes(q,5:6) - objects{p}.boxes(q,3:4));
%         %         end
%         %     end
%         % end
%         % fprintf(fid, '\n');
% 
%         fprintf('%s  --  %g/%g\n',video_name, i, length(dir_list));
%     end
% end
% save(anno_mat_filename, 'annotation');
% % fclose(fid);



%%%%%%%%%%%%%%%%%%%%%%%%%% For val and test set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


dir_list = dir(data_root);
dir_list = dir_list(arrayfun(@(x) x.isdir && ~strcmp(x.name, '.') && ~strcmp(x.name, '..'), dir_list));
for i = 1:length(dir_list)
    video_name = dir_list(i).name;
    data_dir = fullfile(data_root, video_name);
    img_list = dir(fullfile(data_dir, '*.JPEG'));

    % sort frames
    img_list = arrayfun(@(x)x.name, img_list, 'uniformoutput', false);
    img_list_id = zeros(length(img_list), 1);
    for j = 1:length(img_list)
        [ignore, num] = fileparts(img_list{j});
        num = str2double(num);
        img_list_id(j) = num;
    end
    [sortval sortidx] = sort(img_list_id, 'ascend');
    img_list = img_list(sortidx);

    
    % get im_size for test set
    img_size = zeros(length(img_list), 2);
    for j = 1:length(img_list)
        im = imread(fullfile(data_dir, img_list{j}));
        img_size(j, :) = [size(im, 1), size(im, 2)];
    end
    
    
    % % read annotations
    % anno_dir = fullfile(anno_root, video_name);
    % rec = cell(length(img_list), 1);
    % img_size = zeros(length(img_list), 2);
    % for j = 1:length(img_list)
    %     [ignore, filename] = fileparts(img_list{j});
    %     anno_filename = fullfile(anno_dir, [filename '.xml']);
    %     rec{j} = VOCreadxml(anno_filename);
    %     img_size(j, :) = [str2double(rec{j}.annotation.size.height), str2double(rec{j}.annotation.size.width)];
    % end
    % 
    % val_idx = find(cellfun(@(x)isfield(x.annotation,'object'), rec));
    % rec = rec(val_idx);
    % objs = cellfun(@(x)x.annotation.object, rec, 'uniformoutput', false);
    % objs = cat(2,objs{:});
    % obj_idx = unique(arrayfun(@(x)str2double(x.trackid), objs));
    % max_obj = length(obj_idx);
    % 
    % objects = cell(max_obj, 1);
    % obj_category_names = cell(max_obj,1);
    % for j = 1:max_obj
    %     objects{j}.boxes = [];
    %     cur_obj_idx = obj_idx(j);
    %     for k = 1:length(val_idx)
    %         trackid = arrayfun(@(x)str2double(x.trackid), rec{k}.annotation.object);
    %         obj_val_idx = find(cur_obj_idx == trackid);
    %         if ~isempty(obj_val_idx)
    %             if isempty(obj_category_names{j})
    %                 obj_category_names{j} = rec{k}.annotation.object(obj_val_idx).name;
    %             end
    %             %if strcmp(rec{k}.annotation.object(obj_val_idx(1)).name,'n02958343')
    %             box = rec{k}.annotation.object(obj_val_idx).bndbox;
    %             width = str2double(rec{k}.annotation.size.width);
    %             height = str2double(rec{k}.annotation.size.height);
    %             size = [width height width height];
    %             scaled_box = [str2double(box.xmin), str2double(box.ymin), ...
    %                     str2double(box.xmax), str2double(box.ymax)];
    %             scaled_box = bsxfun(@rdivide, scaled_box, size);
    %             objects{j}.boxes = [objects{j}.boxes; j, val_idx(k), scaled_box];
    %         end
    %     end
    % 
    %     label = category_name{strcmp(obj_category_names{j}, WNID)};
    %     objects{j}.obj_idx = j;
    %     objects{j}.start_frame = objects{j}.boxes(1,2);
    %     objects{j}.end_frame = objects{j}.boxes(end,2);
    %     objects{j}.category = label;
    % end
    % annotation{finished_count+1}.obj = objects;
    
    annotation{finished_count+1}.im_list = img_list;
    annotation{finished_count+1}.video_name = sprintf('%s', video_name);
    annotation{finished_count+1}.im_size = img_size;
    finished_count = finished_count + 1;
    fprintf('%s  --  %g/%g\n',video_name, i, length(dir_list));
end

save(anno_mat_filename, 'annotation');

