% D&T training using D model and RPN proposals
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(mfilename('fullpath')), 'startup'));

%% -------------------- CONFIG --------------------
root_path = '../../../dataset/ImageNetVID';

opts.cache_name = 'rfcn_ILSVRC_ResNet101_corr';
conf = rfcn_config_ohem();
conf.root_path = root_path;

% params
opts.cache_name = 'STMN_RFCN';  % STMN_RFCN
opts.debug_set = false;
opts.scan_det = false;
opts.load_scan_det = false;
opts.load_tube = false;
opts.use_track = true;
opts.top_percentage = 0.5; % 0.5
opts.pre_nms_thresh = 0.9; % 0.9
opts.neighbor_rescoring = -1;  % -1 if not doing rescoring, 0.8 in EarlyNbrRescore mode
opts.preload_root = {
                        fullfile(root_path, 'ckpt/rfcn_eval/eval/mat'), ...
                        fullfile(root_path, 'ckpt/stmn_eval/eval/mat'), ...
                    };



% do validation after val_interval iters, or not
opts.do_val                 = true; 
opts.cache_name_            = opts.cache_name;

%% -------------------- TESTING --------------------
%%
% chose test prototxt \in {test_track_bidir.prototxt, test_track_reg.prototxt, 
% test_track_regcls.prototxt,% test_track_regcls_poolreg.prototxt}
% model.test_net_def_file = strrep(model.test_net_def_file,'test_track.prototxt','test_track_reg.prototxt');

conf.sample_vid = true; conf.nFramesPerVid = 2; conf.ims_per_batch = 1 ; conf.regressTracks = true;
conf.bidirTrack = false;
imdb_val_lite = Dataset.ilsvrc15([], 'VIDval16', false, root_path);

% set some paths in a post-hoc way
imdb_val_lite.imdb_test.details.root_dir = root_path;
imdb_val_lite.imdb_test.details.devkit_path = fullfile(root_path, 'devkit');
imdb_val_lite.imdb_test.details.bbox_path = fullfile(root_path, 'Annotations/VID/val');


%% -------------------- TESTING --------------------

if opts.debug_set
    imdb = imdb_val_lite.imdb_test;
    roidb = imdb_val_lite.roidb_test;
    % sample videos
    randseq = randperm(length(imdb.vids));
    % sel_vid_id = randseq(1:5);
    sel_vid_id = [2];
    sel_img_id = find(ismember(imdb.vid_id, sel_vid_id));
    % imdb
    imdb.image_ids = imdb.image_ids(sel_img_id);
    imdb.vids = imdb.vids(sel_vid_id);
    imdb.num_frames = imdb.num_frames(sel_vid_id);
    vid_id = imdb.vid_id(sel_img_id);    
    [ignore, vid_id] = ismember(vid_id, sel_vid_id);
    imdb.vid_id = vid_id;
    imdb.is_blacklisted = imdb.is_blacklisted(sel_img_id);
    imdb.sizes = imdb.sizes(sel_img_id, :);
    imdb.video_ids = imdb.video_ids(sel_vid_id);
    key_arr = {};
    nFrames_arr = {};
    videoSizes_arr = {};
    for idx = 1: length(imdb.vids)
        key = imdb.vids{idx}(1:end-1);
        nFrames_val = imdb.nFrames(key);
        videoSizes_val = imdb.videoSizes(key);
        key_arr{end+1} = key;
        nFrames_arr{end+1} = nFrames_val;
        videoSizes_arr{end+1} = videoSizes_val;
    end
    nFrames = containers.Map(key_arr, nFrames_arr);
    videoSizes = containers.Map(key_arr, videoSizes_arr);
    imdb.nFrames = nFrames;
    imdb.videoSizes = videoSizes;
    % roidb
    roidb.rois = roidb.rois(sel_img_id);
    % modify
    imdb_val_lite.imdb_test = imdb;
    imdb_val_lite.roidb_test = roidb;
end

rfcn_test_vid_preload_det(conf, imdb_val_lite.imdb_test, imdb_val_lite.roidb_test, ...
                  'preload_root',     opts.preload_root, ...
                  'cache_name',       opts.cache_name_,...
                  'write_vid', 0 , ...       
                  'test_classes', [], ...
                  'suffix', '', ...
                  'time_stride', 1, ...
                  'scan_det', opts.scan_det, ...
                  'load_scan_det', opts.load_scan_det, ...
                  'load_tube', opts.load_tube, ...
                  'use_track', opts.use_track, ...
                  'top_percentage', opts.top_percentage, ...
                  'pre_nms_thresh', opts.pre_nms_thresh, ...
                  'neighbor_rescoring', opts.neighbor_rescoring);
exit();






