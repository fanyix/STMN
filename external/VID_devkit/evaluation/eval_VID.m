
function eval_VID(iter_str, res_dir, root_dir, data_dir, write_dir, mat_file)

if isempty(write_dir)
	write_dir = res_dir;
end

if strcmp(mat_file, 'true') == 1 
	pred_file= fullfile(res_dir, sprintf('%d.mat', iter_str));
else
	pred_file= fullfile(res_dir, sprintf('%d.txt', iter_str));
end
fprintf('pred_file=%s', pred_file);
imlist_file = fullfile(res_dir, sprintf('imlist_%d.txt', iter_str));
res_file = fullfile(write_dir, sprintf('res_%d.txt', iter_str));
ground_truth_dir = fullfile(root_dir, 'Annotations/VID/val');
set_file = fullfile(root_dir, 'ImageSets/VID/val.txt');
meta_file = fullfile(data_dir, 'meta_vid.mat');
optional_cache_file = fullfile(data_dir, 'ILSVRC2015_vid_validation_ground_truth_whole.mat');
blacklist_file = '';

[ap recall precision] = eval_vid_detection_by_imlist(pred_file,ground_truth_dir,meta_file,...
    set_file,imlist_file,blacklist_file,optional_cache_file);

% ---------------------
% Print
load(meta_file);
if ~exist(write_dir, 'dir') 
	mkdir(write_dir);
end

% mAP
f=fopen(res_file,'w');
fprintf(f, '-------------\n');
fprintf(f, 'Category\tAP\n');
for i=1:30
	s = synsets(i).name;
	if length(s) < 8
	    fprintf(f, '%s\t\t%0.3f\n',s,ap(i));
	else
	    fprintf(f, '%s\t%0.3f\n',s,ap(i));
	end
end
fprintf(f, ' - - - - - - - - \n');
fprintf(f, 'Mean AP:\t %0.3f\n',mean(ap));
fprintf(f, 'Median AP:\t %0.3f\n',median(ap));
fclose(f);

% Recall
[a,b,c] = fileparts(res_file);
recall_file = fullfile(a, sprintf('recall_%s%s', b, c));
f=fopen(recall_file,'w');
fprintf(f, '-------------\n');
fprintf(f, 'Category\tRecall\n');
recall_arr = [];
for i=1:30
	s = synsets(i).name;
    recall_arr(i) = recall{i}(end);
	if length(s) < 8
	    fprintf(f, '%s\t\t%0.3f\n',s,recall_arr(i));
	else
	    fprintf(f, '%s\t%0.3f\n',s,recall_arr(i));
	end
end
fprintf(f, ' - - - - - - - - \n');
fprintf(f, 'Mean Recall:\t %0.3f\n',mean(recall_arr));
fprintf(f, 'Median Recall:\t %0.3f\n',median(recall_arr));
fclose(f);

