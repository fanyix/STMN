% this script demos the usage of evaluation routines for detection from
% video task.
% the result file 'demo.val.pred.vid.txt' on validation data is evaluated
% against the ground truth.

fprintf('DETECTION FROM VIDEO TASK\n');

% pred_file='demo.val.pred.vid.txt';
pred_file= '/home/SSD2/fanyi-data/ILSVRC2015/res/validation/1.txt';
imlist_file = '/home/SSD2/fanyi-data/ILSVRC2015/res/validation/imlist_1.txt';
meta_file = '../data/meta_vid.mat';
eval_file = '../../ImageSets/VID/val.txt';
blacklist_file = '';
optional_cache_file = '../data/ILSVRC2015_vid_validation_ground_truth_whole.mat';
ground_truth_dir = '/home/SSD2/fanyi-data/ILSVRC2015/Annotations/VID/val';


fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);
fprintf('eval_file: %s\n', eval_file);
fprintf('blacklist_file: %s\n', blacklist_file);
if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

% num_val_files = -1;
% while num_val_files ~= 176126
%     if num_val_files ~= -1
%         fprintf('That does not seem to be the correct directory. Please try again\n');
%     end
%     %ground_truth_dir = input(['Please enter the path to the Validation bounding box ' ...
%     %               'annotations directory: '],'s');
%     val_videos = dir(sprintf('%s/*val*',ground_truth_dir));
%     num_val_files = 0;
%     for i = 1:numel(val_videos)
%         if val_videos(i).isdir
%             val_files = dir(sprintf('%s/%s/*.xml',ground_truth_dir,val_videos(i).name));
%             num_val_files = num_val_files + numel(val_files);
%         end
%     end
% end

% val_videos = dir(sprintf('%s/*val*',ground_truth_dir));
% num_val_files = 0;
% for i = 1:numel(val_videos)
%     if val_videos(i).isdir
%         val_files = dir(sprintf('%s/%s/*.xml',ground_truth_dir,val_videos(i).name));
%         num_val_files = num_val_files + numel(val_files);
%     end
% end


[ap recall precision] = eval_vid_detection_by_imlist(pred_file,ground_truth_dir,meta_file,...
    eval_file,imlist_file,blacklist_file,optional_cache_file);

load(meta_file);
fprintf('-------------\n');
fprintf('Category\tAP\n');
for i=1:30
    s = synsets(i).name;
    if length(s) < 8
        fprintf('%s\t\t%0.3f\n',s,ap(i));
    else
        fprintf('%s\t%0.3f\n',s,ap(i));
    end
end
fprintf(' - - - - - - - - \n');
fprintf('Mean AP:\t %0.3f\n',mean(ap));
fprintf('Median AP:\t %0.3f\n',median(ap));


