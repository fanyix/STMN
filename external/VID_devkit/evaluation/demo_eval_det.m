% this script demos the usage of evaluation routines for detection task
% the result file 'demo.val.pred.det.txt' on validation data is evaluated
% against the ground truth

fprintf('DETECTION TASK\n');

pred_file='demo.val.pred.det.txt';
meta_file = '../data/meta_det.mat';
eval_file = '../ImageSets/DET/val.txt';
blacklist_file = '../data/ILSVRC2015_det_validation_blacklist.txt';
optional_cache_file = '../data/ILSVRC2015_det_validation_ground_truth.mat';

fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);
fprintf('eval_file: %s\n', eval_file);
fprintf('blacklist_file: %s\n', blacklist_file);
if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

num_val_files = -1;
while num_val_files ~= 20121
    if num_val_files ~= -1
        fprintf('That does not seem to be the correct directory. Please try again\n');
    end
    ground_truth_dir = input(['Please enter the path to the Validation bounding box ' ...
                   'annotations directory: '],'s');
    val_files = dir(sprintf('%s/*val*.xml',ground_truth_dir));
    num_val_files = numel(val_files);
end

[ap recall precision] = eval_detection(pred_file,ground_truth_dir,meta_file,eval_file,blacklist_file,optional_cache_file);

load(meta_file);
fprintf('-------------\n');
fprintf('Category\tAP\n');
for i=[1:5 196:200]
    s = synsets(i).name;
    if length(s) < 8
        fprintf('%s\t\t%0.3f\n',s,ap(i));
    else
        fprintf('%s\t%0.3f\n',s,ap(i));
    end
    if i == 5
        fprintf(' ... (190 categories)\n');
    end
end
fprintf(' - - - - - - - - \n');
fprintf('Mean AP:\t %0.3f\n',mean(ap));
fprintf('Median AP:\t %0.3f\n',median(ap));


