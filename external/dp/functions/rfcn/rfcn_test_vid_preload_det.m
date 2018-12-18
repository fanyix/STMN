
function mAP = rfcn_test_vid_preload_det(conf, imdb, roidb, varargin)
% Video-level testing using final D&T model and RPN proposals
% --------------------------------------------------------
% D&T implementation
% Modified from MATLAB  R-FCN (https://github.com/daijifeng001/R-FCN/)
% and Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2017, Christoph Feichtenhofer
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

%% inputs
ip = inputParser;
ip.addRequired('conf',                              @isstruct);
ip.addRequired('imdb',                              @isstruct);
ip.addRequired('roidb',                             @isstruct);
ip.addParamValue('preload_root',    '');
ip.addParamValue('cache_name',      '', 			@isstr);
ip.addParamValue('suffix',          '',             @isstr);
ip.addParamValue('test_classes',    [] 			);
ip.addParamValue('time_stride',   1,          @isscalar);
ip.addParamValue('write_vid',   0,          @isscalar);
ip.addParamValue('scan_det',   false,          @islogical);
ip.addParamValue('load_scan_det',   false,          @islogical);
ip.addParamValue('load_tube',   true,          @islogical);
ip.addParamValue('use_track',   false,          @islogical);
ip.addParamValue('top_percentage',   0.5,          @isscalar);
ip.addParamValue('pre_nms_thresh',   0.9,          @isscalar);
ip.addParamValue('neighbor_rescoring',   -1,          @isscalar);

ip.parse(conf, imdb, roidb, varargin{:});
opts = ip.Results;

%%  set cache dir
cache_dir = fullfile(conf.root_path, 'output', 'rfcn_cachedir', opts.cache_name, imdb.name);
mkdir_if_missing(cache_dir);

%%  init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'log'));
log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
diary(log_file);

num_images = length(imdb.image_ids);
num_classes = imdb.num_classes;

% set random seed
prev_rng = seed_rand(conf.rng_seed);

disp('opts:');
disp(opts);
disp('conf:');
disp(conf);

% heuristic: keep an average of 160 detections per class per images prior to NMS
max_per_set = 160 * num_images;
% heuristic: keep at most 400 detection per class per image prior to NMS
max_per_image = 400;
% detection thresold for each class (this is adaptively set based on the max_per_set constraint)
thresh = -inf * ones(num_classes, 1);
% top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
top_scores = cell(num_classes, 1);
% all detections are collected into:
%    all_boxes[cls][image] = N x 5 array of detections in
%    (x1, y1, x2, y2, score)
aboxes = cell(num_classes, 1);
box_inds = cell(num_classes, 1);

for i = 1:num_classes
    aboxes{i} = cell(length(imdb.image_ids), 1);
    box_inds{i} = cell(length(imdb.image_ids), 1);
end
ascores_track = aboxes{1};
aboxes_track = cell(length(imdb.image_ids), 2);

if isfield(conf,'sample_vid') && conf.sample_vid
    [inds_val,frames_val] = video_generate_random_minibatch([], [], imdb, conf, false);
    % frames_val = cellfun(@(x) padarray(x,conf.nFramesPerVid-mod(length(x),conf.nFramesPerVid),'symmetric','post'), frames_val, 'UniformOutput' , false); % sample first frames
    frames_val = cellfun(@(x) padarray(x,mod(length(x),conf.nFramesPerVid),'symmetric','post'), frames_val, 'UniformOutput' , false); % sample first frames
    imdb.classes = cellfun(@(x) strrep(x, '_',' '),imdb.classes, 'uniform',0);
end
if ~isfield(conf,'nFramesPerVid') 
  conf.nFramesPerVid = 1;
end
t_stride = opts.time_stride ; testThis = true;
count = 0;

% create lock dir
scan_cache_dir = fullfile(cache_dir, 'scan_cache');
lock_dir = fullfile(cache_dir, 'lock');
if opts.scan_det
    if ~exist(lock_dir, 'dir')
        mkdir(lock_dir);
    end
    if ~exist(scan_cache_dir, 'dir')
        mkdir(scan_cache_dir)
    end
end

if ~exist(fullfile(cache_dir, ['tracks_' imdb.name '.mat']), 'file')    
    if ~opts.load_scan_det
        for v = 1:numel(inds_val)        
          lock_file = fullfile(lock_dir, fileparts(imdb.vids{v}));
          if ~opts.scan_det || ~exist(lock_file, 'file')
              % create the lock
              if opts.scan_det
                  cmd = sprintf('touch %s', lock_file);
                  unix(cmd);
                  aboxes = cell(num_classes, 1);
                  box_inds = cell(num_classes, 1);
                  for i = 1:num_classes
                      aboxes{i} = cell(length(imdb.image_ids), 1);
                      box_inds{i} = cell(length(imdb.image_ids), 1);
                  end
                  ascores_track = aboxes{1};
                  aboxes_track = cell(length(imdb.image_ids), 2);
              end

              for kk = 1:t_stride:numel(frames_val{v})-t_stride
                  sub_db_inds = frames_val{v}(kk:kk+t_stride);
                  sub_db_inds = sub_db_inds(round(linspace(1, length(sub_db_inds), conf.nFramesPerVid)));
                  d = roidb.rois(sub_db_inds);
                  if ~isempty(opts.test_classes)
                    testThis = false;
                    for ii = 1:length(opts.test_classes)
                      if any(opts.test_classes(ii) == d(1).class), testThis = true; end
                    end
                    if ~testThis, continue; end
                  end
                  count = count + 1;
                  fprintf('%s: test (%s) vid %d/%d frame %d/%d\n', procid(), imdb.name, v,numel(inds_val), kk,numel(frames_val{v}));

                  [image_roidb_val, ~, ~] = rfcn_prepare_image_roidb_offline(conf, imdb, roidb.rois(sub_db_inds) , conf.root_path, sub_db_inds);
                  boxes = {}; scores = {}; mot_boxes = {}; skip = false;
                  for b = 1:numel(sub_db_inds)
                      % local params for detection reading
                      tiny_box_size = 5;
                      max_roi_num = 5000;
                      % parsing
                      parts = split(image_roidb_val(b).image_id, '/');
                      vid_name = parts{1};
                      frm_idx = str2double(parts{2}) + 1; % +1 is for the starting index offset
                      % here cur_mot_boxes is just a copy of cur_boxes, no motion prediction at all
                      [cur_boxes, cur_scores, cur_mot_boxes] = preload_det(vid_name, frm_idx, imdb.classes, opts.preload_root);
                      % check
                      if isempty(cur_boxes) || isempty(cur_scores) 
                        skip = true;
                        continue;
                      end
                      % prune out tiny boxes
                      wid = cur_boxes(:,3) - cur_boxes(:,1) + 1;
                      hgt = cur_boxes(:,4) - cur_boxes(:,2) + 1;
                      non_tiny_idx = wid>=tiny_box_size & hgt>=tiny_box_size;
                      cur_boxes = cur_boxes(non_tiny_idx, :);
                      cur_scores = cur_scores(non_tiny_idx, :);
                      cur_mot_boxes = cur_mot_boxes(non_tiny_idx, :);
                      % check again
                      if isempty(cur_boxes) || isempty(cur_scores) 
                        skip = true;
                        continue;
                      end

                      %%%%%%%%%%%%% aggregate by per-category averaging %%%%%%%%%%%%%
                      % agg_top_K = 3;
                      % [agg_boxes, IA, IC] = unique(cur_boxes, 'rows');
                      % agg_scores = zeros(size(agg_boxes, 1), size(cur_scores, 2));
                      % for ii = 1: numel(IA)
                      %     val_indicator = IC == ii;
                      %     ii_scores = cur_scores(val_indicator, :);
                      %     sort_ii_scores = sort(ii_scores, 1, 'descend');
                      %     sort_ii_scores = sort_ii_scores(1: min(size(sort_ii_scores, 1), agg_top_K), :);
                      %     agg_scores(ii,:) = mean(sort_ii_scores, 1);
                      % end
                      % cur_boxes = agg_boxes;
                      % cur_scores = agg_scores;
                      %%%%%%%%%%%%% aggregate by averaging %%%%%%%%%%%%%
                      % agg_top_K = 30;
                      % max_pos_score = max(cur_scores, [], 2);
                      % [agg_boxes, IA, IC] = unique(cur_boxes, 'rows');
                      % agg_scores = zeros(size(agg_boxes, 1), size(cur_scores, 2));
                      % for ii = 1: numel(IA)
                      % val_indicator = IC == ii;
                      % ii_scores = cur_scores(val_indicator, :);
                      % ii_max_pos_score = max_pos_score(val_indicator);
                      % [ignore, sort_idx] = sort(ii_max_pos_score, 'descend');
                      % sort_idx = sort_idx(1: min(numel(sort_idx), agg_top_K));
                      % agg_scores(ii,:) = mean(ii_scores(sort_idx, :), 1);
                      % end
                      % cur_boxes = agg_boxes;
                      % cur_scores = agg_scores;
                      %%%%%%%%%%%%% per-category nms %%%%%%%%%%%%%
                      % nms_idx = [];
                      % for cls_idx = 1:size(cur_scores, 2)
                      % cur_nms_idx = nms([cur_boxes, cur_scores(:,cls_idx)], opts.pre_nms_thresh);
                      % nms_idx = [nms_idx; cur_nms_idx];
                      % end
                      % nms_idx = unique(nms_idx);
                      %%%%%%%%%%%%% max-pos-score nms %%%%%%%%%%%%%
                      max_pos_score = max(cur_scores, [], 2);
                      nms_idx = nms([cur_boxes, max_pos_score], opts.pre_nms_thresh);
                      cur_boxes = cur_boxes(nms_idx, :);
                      cur_scores = cur_scores(nms_idx, :);
                      cur_mot_boxes = cur_mot_boxes(nms_idx, :);
                      
                      % extra caution
                      if size(cur_boxes, 1) > max_roi_num
                          max_pos_score = max(cur_scores, [], 2);
                          [a, b] = sort(max_pos_score, 'descend');
                          b = b(1: max_roi_num);
                          cur_boxes = cur_boxes(b, :);
                          cur_scores = cur_scores(b, :);
                          cur_mot_boxes = cur_mot_boxes(b, :);
                      end

                      % re-scoring
                      if opts.neighbor_rescoring >= 0
                        rescores = cur_scores;
                        o = boxoverlap(cur_boxes, cur_boxes);
                        for oi = 1:size(o, 1)
                            nbr_idx = o(oi, :) >= opts.neighbor_rescoring;
                            if sum(nbr_idx) > 0
                                nbr_ov = o(oi, nbr_idx);
                                nbr_ov = reshape(nbr_ov, [1, numel(nbr_ov)]);
                                tmp_row = nbr_ov * cur_scores(nbr_idx, :);
                                tmp_row = tmp_row / (sum(nbr_ov) + eps);
                                rescores(oi, :) = tmp_row;
                            end
                        end
                        cur_scores = rescores;
                      end

                      boxes{end+1} = cur_boxes;
                      scores{end+1} = cur_scores;
                      mot_boxes{end+1} = cur_mot_boxes;
                  end
                  if skip 
                      parts = split(image_roidb_val(1).image_id, '/');
                      vid_name = parts{1};
                      frm_idx = parts{2};
                      fprintf('WARNING: skip %s frame %s\n', vid_name, frm_idx);
                      continue;
                  end

                  % recover the preload detections
                  % track_boxes = boxes(1);
                  mot_valid = mot_boxes{1}(:, 5);
                  track_boxes = {mot_boxes{1}(:, 1:4)};
                  boxes_frames = cell(length(boxes), 1);
                  scores_frames = cell(length(boxes), 1);
                  for cur_idx = 1: length(boxes)
                    boxes_frames{cur_idx} = repmat(boxes{cur_idx}, [1, length(imdb.classes)]);
                    scores_frames{cur_idx} = scores{cur_idx};
                  end

                  %% boost track boxes
                  if ~iscell(track_boxes), track_boxes = {track_boxes}; end;
                  for i=1:numel(track_boxes)
                    tracklets = find(max(scores_frames{i}, [], 2)  > 0.01);
                    if any(tracklets)
                      ascores_track{sub_db_inds(1)} = [ascores_track{sub_db_inds(1)}; scores_frames{i}(tracklets,:)] ;
                      if i==1 % forward track
                        aboxes_track{sub_db_inds(1),1} = [aboxes_track{sub_db_inds(1),1}; boxes_frames{i}(tracklets,1:4)];
                        aboxes_track{sub_db_inds(1),2}  = [aboxes_track{sub_db_inds(1),2}; track_boxes{i}(tracklets,1:4)];
                      else % backward track
                        aboxes_track{sub_db_inds(1),1}  = [aboxes_track{sub_db_inds(1),1}; track_boxes{i}(tracklets,1:4)];
                        aboxes_track{sub_db_inds(1),2}  = [aboxes_track{sub_db_inds(1),2}; boxes_frames{i}(tracklets,1:4)];
                      end
                    end
                  end

                  %% resort frame boxes
                  count2 = 0;
                  for i=sub_db_inds(:)'
                    count2 = count2+1;
                    boxes = boxes_frames{count2};  scores = scores_frames{count2};
                    for j = 1:num_classes
                        inds = find(scores(:, j) > thresh(j));
                        if ~isempty(inds)
                            [~, ord] = sort(scores(inds, j), 'descend');
                            ord = ord(1:min(length(ord), max_per_image));
                            inds = inds(ord); 
                            cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                            cls_scores = scores(inds, j); bg_scores = 1-sum(scores(inds,:),2);
                            aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores), single(bg_scores))];
                            if ~isempty(box_inds{j}{i}), inds=  inds + numel(box_inds{j}{i}); end
                            box_inds{j}{i} = [box_inds{j}{i}; inds ];
                        else
                            aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 6, 'single')];
                            box_inds{j}{i} = box_inds{j}{i};
                        end
                    end
                  end
                  if mod(count, 1000) == 0 && ~opts.scan_det
                    for j = 1:num_classes
                        [aboxes{j}, box_inds{j}, thresh(j)] = keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
                    end
                    disp(thresh);
                  end
              end
              % save cache
              if opts.scan_det
                cache_file = fullfile(scan_cache_dir, sprintf('%s.mat', fileparts(imdb.vids{v})));
                save(cache_file, 'aboxes', 'box_inds', 'aboxes_track', 'ascores_track');
              end
          end
        end

        if opts.scan_det
            return;
        end
    else
        % load scan det
        scan_det_load_tic = tic;
        scan_det_load_count = 0;
        for v = 1:numel(inds_val)    
            cache_file = fullfile(scan_cache_dir, sprintf('%s.mat', fileparts(imdb.vids{v})));
            res = load(cache_file);
            cur_aboxes = res.aboxes;
            cur_box_inds = res.box_inds;
            cur_aboxes_track = res.aboxes_track;
            cur_ascores_track = res.ascores_track;
            for ii = (scan_det_load_count+1):numel(cur_aboxes{1})
                if ~isempty(cur_aboxes_track{ii, 1})
                    assert(isempty(aboxes_track{ii, 1}));
                    assert(isempty(aboxes_track{ii, 2}));
                    assert(isempty(ascores_track{ii}));
                    aboxes_track{ii, 1} = cur_aboxes_track{ii, 1};
                    aboxes_track{ii, 2} = cur_aboxes_track{ii, 2};
                    ascores_track{ii} = cur_ascores_track{ii};
                end
                for c = 1: num_classes
                    if ~isempty(cur_aboxes{c}{ii})
                        assert(isempty(aboxes{c}{ii}));
                        assert(isempty(box_inds{c}{ii}));
                        aboxes{c}{ii} = cur_aboxes{c}{ii};
                        box_inds{c}{ii} = cur_box_inds{c}{ii};
                        scan_det_load_count = max(scan_det_load_count, ii);
                    end
                end
                % compute threshold
                if mod(scan_det_load_count, 5000) == 0
                    for j = 1:num_classes
                        [aboxes{j}, box_inds{j}, thresh(j)] = keep_top_k(aboxes{j}, box_inds{j}, scan_det_load_count, max_per_set, thresh(j));
                    end
                    disp(thresh);
                end
            end 
            scan_det_load_time = toc(scan_det_load_tic);
            fprintf('%d/%d video loaded, %.3f sec.\n', v, numel(inds_val), scan_det_load_time/v);
        end
    end

    for j = 1:num_classes
        [aboxes{j}, box_inds{j}, thresh(j)] = keep_top_k(aboxes{j}, box_inds{j}, numel(aboxes{j}), max_per_set, thresh(j));
    end
    disp(thresh);

    for i = 1:num_classes
        top_scores{i} = sort(top_scores{i}, 'descend');  
        if (length(top_scores{i}) > max_per_set)
            thresh(i) = top_scores{i}(max_per_set);
        end
        % go back through and prune out detections below the found threshold
        for j = 1:length(imdb.image_ids)
            if ~isempty(aboxes{i}{j})
                I = find(aboxes{i}{j}(:,5) < thresh(i));
                aboxes{i}{j}(I,:) = [];
                box_inds{i}{j}(I,:) = [];
            end
        end
        save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
        boxes = aboxes{i};
        inds = box_inds{i};
        save(save_file, 'boxes', 'inds');
        clear boxes inds;
    end
    save_file = fullfile(cache_dir, ['tracks_' imdb.name opts.suffix]);
    save(save_file, 'aboxes_track', 'ascores_track');
    rng(prev_rng);
else
    imdb.classes = cellfun(@(x) strrep(x, '_',' '),imdb.classes, 'uniform',0);
    for i = 1:num_classes
        load(fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]));
        aboxes{i} = boxes;
    end
    load(fullfile(cache_dir, ['tracks_' imdb.name opts.suffix]));
end

if ~opts.use_track
    % clean out all the track info
    for idx = 1: size(aboxes_track, 1)
        aboxes_track{idx, 1} = [];
        aboxes_track{idx, 2} = [];
        ascores_track{idx} = [];
    end
end

% generate object tracks
track_boxes = repmat({cell(size(aboxes{1}))},size(aboxes));
[inds_val,frames_val] = video_generate_random_minibatch([], [], imdb, conf, false);

save_file = fullfile(cache_dir, ['tubes_' imdb.name opts.suffix '.mat']);
paths_loaded = false;

if opts.load_tube && exist(save_file, 'file')               
    paths_loaded = true;
    res = load(save_file);
    paths = res.paths;
    remainBoxes = res.remainBoxes;
else
    paths = cell(numel(inds_val),numel(aboxes));
    remainBoxes = cell(numel(inds_val),numel(aboxes));
end

for v = 1:numel(inds_val)
  tic_toc_print('generating tracks for video: %d/%d\n', v, length(inds_val));  
  for c = 1:numel(aboxes)
    frameIdx = frames_val{v};
    frameBoxes =  aboxes{c}(frameIdx);
    nonempty_frames = (~cellfun(@isempty,frameBoxes));
    if any(nonempty_frames)
        frameBoxes = frameBoxes(nonempty_frames);
        frameIdx = frameIdx(nonempty_frames);
        if ~paths_loaded
            [paths{v,c}, remainBoxes{v,c}] = make_tubes(frameBoxes, 25, true, {aboxes_track(frameIdx,:), ascores_track(frameIdx), c} ); 
        end
        % tweak the score
        path_scores = cell(length(paths{v,c}), 1);
        for idx = 1:numel(paths{v,c})
            [top_scores] = sort(paths{v,c}(idx).boxes(:, end), 'descend'); 
            mean_top_scrs = mean(top_scores(1:ceil(numel(top_scores) * opts.top_percentage)));
            path_scores{idx} = paths{v,c}(idx).boxes(:, end) + 2*mean_top_scrs;
        end
        for j=1:numel(paths{v,c})
            for k=1:numel(frameIdx)
                track_boxes{c}{frameIdx(k)} = [track_boxes{c}{frameIdx(k)}; [paths{v,c}(j).boxes(k,1:4) path_scores{j}(k)]];
            end
        end

        % add the left-out detections back
        for k=1:numel(frameIdx)
           if ~isempty(remainBoxes{v,c}) && ~isempty(remainBoxes{v,c}{k})
               track_boxes{c}{frameIdx(k)} = [track_boxes{c}{frameIdx(k)}; remainBoxes{v,c}{k}];
           end
        end
    end
  end
end
if ~paths_loaded
    save(save_file, 'paths', 'remainBoxes');
end    
    
% ------------------------------------------------------------------------
% Peform AP evaluation
% ------------------------------------------------------------------------

if isequal(imdb.eval_func, @imdb_eval_voc)
    new_parpool();
    parfor model_ind = 1:num_classes
      cls = imdb.classes{model_ind};
      res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
    end
else
    res = imdb.eval_func(track_boxes, imdb, conf, opts.suffix, false);
    save(fullfile(cache_dir, 'AP.mat'), 'res');
end

if ~isempty(res)
    fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
    fprintf('Results:\n');
    aps = [res(:).ap]' * 100;
    disp(aps);
    disp(mean(aps));
    fprintf('~~~~~~~~~~~~~~~~~~~~\n');
    mAP = mean(aps);
else
    mAP = nan;
end
    
end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,5), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,5) >= thresh);
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end

function [boxes, scores, mot_boxes] = preload_det(vid_name, frm_idx, cls_list, preload_root)
    preload_cls_list = {'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', ...
                'dog', 'domestic cat', 'elephant', 'fox', 'giant panda', 'hamster', 'horse', 'lion', ...
                'lizard', 'monkey', 'motorcycle', 'rabbit', 'red panda', 'sheep', 'snake', 'squirrel', ...
                'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra'
    };
    [a, b] = ismember(cls_list, preload_cls_list);

    boxes = [];
    scores = [];
    mot_boxes = [];
    
    if ischar(preload_root)
        det_filename = fullfile(preload_root, vid_name, sprintf('%d.mat', frm_idx));
        if exist(det_filename, 'file')
            x = load(det_filename);
            x = x.x;
            boxes = x(:, 1:4);
            scores = x(:, 5:end);
            scores = scores(:, b);
            % motion
            mot_filename = fullfile(preload_root, vid_name, sprintf('mot_%d.mat', frm_idx));
            if exist(mot_filename, 'file')
                x = load(mot_filename);
                x = x.x;
                mot_boxes = [x ones(size(x, 1), 1)];
            else
                mot_boxes = [boxes zeros(size(boxes, 1), 1)];
            end
        end
    else
        for idx = 1: length(preload_root)
            det_filename = fullfile(preload_root{idx}, vid_name, sprintf('%d.mat', frm_idx));
            if exist(det_filename, 'file')
                x = load(det_filename);
                x = x.x;
                cur_boxes = x(:, 1:4);
                cur_scores = x(:, 5:end);
                cur_scores = cur_scores(:, b);
                boxes = [boxes; cur_boxes];
                scores = [scores; cur_scores];
                % motion
                mot_filename = fullfile(preload_root{idx}, vid_name, sprintf('mot_%d.mat', frm_idx));
                if exist(mot_filename, 'file')
                    x = load(mot_filename);
                    x = x.x;
                    mot_boxes = [mot_boxes; [x ones(size(x, 1), 1)]];
                else
                    mot_boxes = [mot_boxes; [cur_boxes zeros(size(cur_boxes, 1), 1)]];
                end
            end
        end
    end
end

























