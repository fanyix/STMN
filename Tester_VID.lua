--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
------------------------------------------------------------------------------]]

require 'sys'
local mu = require 'myutils'
local utils = paths.dofile('utils.lua')
local tds = require 'tds'
local hdf5 = require 'hdf5'
local matio = require 'matio'
local Tester = torch.class('fbcoco.Tester_VID')
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
local cur_dir = paths.dirname(paths.thisfile(nil))

function Tester:__init(module, transformer, loader, scale, max_size, opt)
   self.module = module
   self.transformer = transformer
   self.loader = loader
   if module and transformer then
      self.detec = fbcoco.ImageDetect(self.module, self.transformer, scale, max_size)
   end
   self.nms_thresh = opt.test_nms_threshold or 0.3   
   self.test_epoch = opt.test_epoch
   self.eval_res_dir = opt.eval_res_dir
   self.inference_method = opt.inference_method or 'static'
   self.batch_T = opt.timestep_per_batch
   self.evaluate_on_the_fly = opt.evaluate_on_the_fly or false
   self.eval_iou_thresh = opt.eval_iou_thresh or 0.5
   self.save_mat_file = opt.save_mat_file or false
   self.save_MatchTransCoef = opt.save_MatchTransCoef or false
   self.remove_near_dup = opt.remove_near_dup or false
   self.threads = Threads(10,
   function()
      require 'torch'
   end)
   if module then
      module:apply(function(m)
         if torch.type(m) ==  'nn.DataParallelTable' then
            self.data_parallel_n = #m.gpuAssignments
         end
      end)
      print('data_parallel_n', self.data_parallel_n)
      -- to determine num of output classes
      local im_num = opt.seq_per_batch * opt.timestep_per_batch
      
      local param, grad = module:parameters()
      local input_dim = param[1]:size(2)
      local input = {torch.CudaTensor(im_num, input_dim, 224, 224),
                    torch.Tensor{1, 1, 1, 100, 100}:view(1, 5):expand(im_num, 5):cuda()}
      local default_N, default_T = utils.set_NT(module, 1, opt.timestep_per_batch)  
      module:forward(input)
      utils.set_NT(module, default_N, default_T)
      self.num_classes = module.output[1]:size(2) - 1
      self.thresh = torch.ones(self.num_classes):mul(opt.score_threshold or -1)
   end
   
   -- Control vars for full evaluation mode
   self.mirror = nil  -- 'FRONT_COPY', 'COPY' or 'FLIP', set to nil when not using mirroring
   self.mirror_min_len = 2  -- pad + 1
   if opt.model == 'stmn' then
      self.DETECTOR = self.detec.detect_VID_CENTER_OVERLAP   -- Use detect_VID_SEQ for RFCN, 
                                                             -- or {detect_VID_CENTER, *detect_VID_CENTER_OVERLAP, 
                                                             -- detect_VID_CENTER_OVERLAP_MULTISCALE, detect_VID_LONGMEM, 
                                                             -- detect_VID_MULWINSIZE, detect_VID_LAST} for STMN
   elseif opt.model == 'rfcn' then
      self.DETECTOR = self.detec.detect_VID_SEQ
   else
      assert(false, 'Unknown model type.')
   end
end

function Tester:writetxt_box(cls_boxes, cls_idx, glb_idx, eval_filename)
  -- write results into text files to be used by ILSVRC eval kit
  -- [img_ids obj_labels obj_confs xmin ymin xmax ymax]
  local entry
  local full_string = ''
  for entry_idx = 1, cls_boxes:size(1) do
    local box = cls_boxes[{entry_idx, {1, 4}}]
    local score = cls_boxes[{entry_idx, 5}]
    entry = string.format('%d %d %.4f %d %d %d %d', glb_idx, cls_idx, score, 
            box[1], box[2], box[3], box[4])
    full_string = full_string .. entry .. '\n'
  end
  
  local file = io.open(eval_filename, "a+")
  file:write(full_string)
  io.close(file)
  --print('Write finished.')
end

function Tester:test(iter, res_dir, root_dir, evalkit_dir, evalkit_data_dir, randSeed)
   self.module:evaluate()
   
   -- set a temporary random seed
   randSeed = randSeed or 981
   local rngState = torch.getRNGState()
   torch.manualSeed(randSeed)
   
   -- save video_index_list and video_ptr of the loader
   local video_index_list = self.loader.video_index_list:clone()
   local video_ptr = self.loader.video_ptr
   self.loader:shuffle()
   local non_redundant_sampling = self.loader.non_redundant_sampling
   self.loader.non_redundant_sampling = false
   
   local aboxes_t = tds.hash()
   local raw_output = tds.hash()
   local raw_bbox_pred = tds.hash()
   local eval_filename = paths.concat(res_dir, string.format('%g.txt', iter))
   local imlist_filename = paths.concat(res_dir, string.format('imlist_%g.txt', iter))
   paths.mkdir(paths.dirname(eval_filename))
   
   -- load image and test
   local video_n = #self.loader.video_names
   local batch_n = self.loader.batch_N
   local inner_loop_size = math.ceil(video_n / batch_n)
   local glb_idx_coll = {}
   local timer = torch.Timer()
   local eval_count = 0
   local sample_record = {}
   timer:reset()
   
   -- erase any evaluation record 
   self.eval_stat = nil
   self.gt_record = nil
   
   for epoch = 1, self.test_epoch do
    for iter = 1, inner_loop_size do
      local glb_idx, _, record = self:evalOne(eval_filename)
      -- write down sample record
      table.insert(glb_idx_coll, glb_idx)
      for vid_name, frm_idx in pairs(record.frm) do
        if not sample_record[vid_name] then
          sample_record[vid_name] = {}
        end
        for idx, frm in ipairs(frm_idx:totable()) do
          sample_record[vid_name][frm] = true
        end
      end
      -- profiling
      eval_count = eval_count + batch_n
      local time_elapsed = timer:time().real
      print(string.format('%g/%g video processed, %.3f sec/vid', eval_count, 
          self.test_epoch*video_n, time_elapsed/eval_count))
    end
   end
   
   if self.evaluate_on_the_fly then
     -- calculate total number of object instances, across all frames/videos
     for vid_name, frm_list in pairs(sample_record) do
      local vid_frm_list = torch.FloatTensor(mu.keys(frm_list))
      for obj_idx, obj in ipairs(self.loader.anno[vid_name].obj) do
        if not self.eval_stat[obj.category].total_obj then
          self.eval_stat[obj.category].total_obj = 0
        end
        local val_idx = torch.cmul(vid_frm_list:ge(obj.start_frame), vid_frm_list:le(obj.end_frame))
        self.eval_stat[obj.category].total_obj = self.eval_stat[obj.category].total_obj + torch.sum(val_idx)
      end
     end     
     -- compute mAP
     local mAP = {}
     local str = {'---------------'}
     table.insert(str, 'Category\t\tAP')
     for cat_name, res in pairs(self.eval_stat) do
      local check_map = torch.cat(res.check_map, 1)
      local y, i = torch.sort(check_map:select(2, 2), 1, true)
      local sorted_check_col = check_map:select(2, 1):index(1, i)
      local cumsum = torch.cumsum(sorted_check_col, 1)
      local recall = torch.div(cumsum, res.total_obj+1e-8)
      local precision = torch.cdiv(cumsum, torch.range(1, sorted_check_col:nElement()):float())
      local AP = self:VOCap(recall, precision)
      table.insert(mAP, AP)
      if string.len(cat_name) < 8 then
        table.insert(str, string.format('%s\t\t\t%0.3f', cat_name, AP))
      else
        table.insert(str, string.format('%s\t\t%0.3f', cat_name, AP))
      end
     end
     mAP = torch.mean(torch.FloatTensor(mAP))
     table.insert(str, '---------------')
     table.insert(str, string.format('mAP\t\t\t%0.3f', mAP))
     -- concat the string 
     str = table.concat(str, '\n')
     local OTF_eval_filename = paths.concat(res_dir, string.format('OTF_%g.txt', iter))
     local f = io.open(OTF_eval_filename, 'w+')
     f:write(str)
     io.close(f)
   else
     -- write imlist file
     local glb_idx = torch.cat(glb_idx_coll, 1)
     glb_idx = glb_idx:totable()
     local glb_idx_str = table.concat(glb_idx, ' ')
     local f = io.open(imlist_filename, 'w+')
     f:write(glb_idx_str)
     io.close(f)
     
     -- evoke evaluation on the written results
     res_dir = res_dir or paths.concat(cur_dir, '../dataset/vis/quant/')
     root_dir = root_dir or paths.concat(cur_dir, '../dataset/')
     evalkit_data_dir = evalkit_data_dir or paths.concat(cur_dir, 'external/VID_devkit_data/')
     evalkit_dir = evalkit_dir or paths.concat(cur_dir, 'external/VID_devkit/evaluation/')
     local matlab_loc = 'matlab'
     local exec_cmd = 
          string.format('%s -nodesktop -nodisplay -r "cd %s; eval_VID(%d, \'%s\', \'%s\', \'%s\', \'%s\', \'false\');exit;"', 
          matlab_loc, evalkit_dir, iter, res_dir, root_dir, evalkit_data_dir, res_dir)
     --os.execute(exec_cmd)
     print('Start quant evaluation ...')
     local handle = io.popen(exec_cmd)
     local result = handle:read("*a")
     handle:close()
     print('Quant evaluation done.')
   end
   
   -- set back video_index_list and video_ptr of the loader
   self.loader.video_index_list:copy(video_index_list)
   self.loader.video_ptr = video_ptr
   self.loader.non_redundant_sampling = non_redundant_sampling
   
   -- set back the random seed
   torch.setRNGState(rngState)
   
   self.module:training()
end

function Tester:test_FULL(tag, res_dir)
   self.module:evaluate()   
   
   local aboxes_t = tds.hash()
   local raw_output = tds.hash()
   local raw_bbox_pred = tds.hash()
   local eval_filename = paths.concat(res_dir, string.format('%s.txt', tag))
   local imlist_filename = paths.concat(res_dir, string.format('imlist_%s.txt', tag))
   paths.mkdir(paths.dirname(eval_filename))
   
   -- load image and test
   local video_n = #self.loader.video_names
   local glb_idx_coll = {}
   local timer = torch.Timer()
   timer:reset()
   
   -- init the lock dir and count variable
   local lock_dir = paths.concat(res_dir, 'lock')
   local count = 0
   
   -- erase evalutation statistics
   self.eval_stat = nil
   self.gt_record = nil
   self.glb_map = nil
   
   for video_idx = 1, video_n do
      local video_name = self.loader.video_names[video_idx]
      -- check lock
      local lock_filename = paths.concat(lock_dir, string.format('%s', video_name))
      paths.mkdir(paths.dirname(lock_filename))
      if not paths.filep(lock_filename) then
        -- write lock
        local fd = io.open(lock_filename, 'w')
        fd:write('lock')
        fd:close()
        -- evaluate
        local glb_idx, _, chains = self:evalOne_FULL(eval_filename, video_name)
        table.insert(glb_idx_coll, glb_idx)
        count = count + 1
        local time_elapsed = timer:time().real
        print(string.format('%g/%g video processed, %.3f sec/vid', video_idx, 
            video_n, time_elapsed/count))
      end
   end
   
   -- save a mapping from global image id to idx
   local str = {}
   for glb_img_id, glb_img_idx in pairs(self.glb_map.name_to_idx) do
    local cur_str = string.format('%s %d', glb_img_id, glb_img_idx)
    table.insert(str, cur_str)
   end 
   str = table.concat(str, '\n')
   local glb_map_filename = paths.concat(res_dir, string.format('map_%s.txt', tag))
   local f = io.open(glb_map_filename, 'w+')
   f:write(str)
   io.close(f)
     
   
   if self.evaluate_on_the_fly then
     -- calculate total number of object instances, across all frames/videos
     for vid_name, vid in pairs(self.loader.anno) do
      for obj_idx, obj in ipairs(vid.obj) do
        if not self.eval_stat[obj.category].total_obj then
          self.eval_stat[obj.category].total_obj = 0
        end
        self.eval_stat[obj.category].total_obj = self.eval_stat[obj.category].total_obj + 
                                                  (obj.end_frame - obj.start_frame + 1)
      end
     end
     -- calculate mAP   
     local mAP = {}
     local str = {'---------------'}
     table.insert(str, 'Category\tAP')
     for cat_name, res in pairs(self.eval_stat) do
      local check_map = mu.safe_concat(res.check_map, 1)
      local y, i = torch.sort(check_map:select(2, 2), 1, true)
      local sorted_check_col = check_map:select(2, 1):index(1, i)
      local cumsum = torch.cumsum(sorted_check_col, 1)
      local recall = torch.div(cumsum, res.total_obj+1e-8)
      local precision = torch.cdiv(cumsum, torch.range(1, sorted_check_col:nElement()):float())
      local AP = self:VOCap(recall, precision)
      table.insert(mAP, AP)
      if string.len(cat_name) < 8 then
        table.insert(str, string.format('%s[%d]\t\t%0.3f', cat_name, res.total_obj, AP))
      else
        table.insert(str, string.format('%s[%d]\t%0.3f', cat_name, res.total_obj, AP))
      end
     end
     mAP = torch.mean(torch.FloatTensor(mAP))
     table.insert(str, '---------------')
     table.insert(str, string.format('mAP\t\t%0.3f', mAP))
     -- concat the string 
     str = table.concat(str, '\n')
     local OTF_eval_filename = paths.concat(res_dir, string.format('OTF.txt'))
     local f = io.open(OTF_eval_filename, 'w+')
     f:write(str)
     io.close(f)
   else
     -- write imlist file
     local glb_idx = torch.cat(glb_idx_coll, 1)
     glb_idx = glb_idx:totable()
     local glb_idx_str = table.concat(glb_idx, ' ') .. ' '  -- this trick simplifies file concatenation 
     local f = io.open(imlist_filename, 'w+')
     f:write(glb_idx_str)
     io.close(f)
   end
   
   self.module:training()
end

function Tester:VOCap(rec, prec)
  local mrec = rec.new(rec:nElement()+2)
  mrec[1] = 0
  mrec[mrec:nElement()] = 1
  mrec:narrow(1, 2, mrec:nElement()-2):copy(rec)
  
  local mpre = prec.new(prec:nElement()+2)
  mpre[1] = 0
  mpre[mpre:nElement()] = 0
  mpre:narrow(1, 2, mpre:nElement()-2):copy(prec)
  
  local ap = 0
  local N = mpre:nElement()
  for _, i in ipairs(torch.range(N-1, 1, -1):totable()) do
    mpre[i] = math.max(mpre[i], mpre[i+1])
  end
  
  local idx = torch.nonzero((mrec[{{2, N}}]-mrec[{{1, N-1}}]):ne(0))
  if idx:nElement() > 0 then
    idx = idx:view(-1) + 1
    idx = idx:long()
    ap = torch.sum(torch.cmul((mrec:index(1, idx) - mrec:index(1, idx-1)), mpre:index(1, idx)))
  end
  return ap
end


function Tester:evalOne_FULL(eval_filename, video_name)
  collectgarbage()
  local thresh = self.thresh
  local frame_stride = self.loader.frame_stride
  local pad = (self.batch_T - 1) / 2
  
  assert(self.mirror_min_len > 1, 'mirror_min_len must be at least 2.')
  local glb_idx_coll, img_boxes_coll, feat_coll, detailed_boxes_coll = {}, {}, {}, {}
  -- get the video length
  local video_len = #self.loader.anno[video_name].im_list
  
  for start_idx = 1, math.min(frame_stride, video_len) do      
    local frame_seq = torch.range(start_idx, video_len, frame_stride)
    
    -- For now let's just assume we can always load #frame_seq frames into cpu memory    
    local images, boxes = self.loader:light_sample_target(video_name, frame_seq)
    local T, H, W = images:size(1), images:size(3), images:size(4)
    
    if self.mirror and T >= self.mirror_min_len then
      assert(self.batch_T % 2 == 1)
      
      -- front
      local front_fill_idx
      if self.mirror == 'FLIP' then
        front_fill_idx = torch.range(math.min(pad + 1, T), 2, -1)
        local fill_idx_len = front_fill_idx:nElement()
        if fill_idx_len < pad then
          front_fill_idx = torch.cat({torch.Tensor(pad - fill_idx_len):fill(T), front_fill_idx}, 1)
        end
      elseif self.mirror == 'COPY' then
        front_fill_idx = torch.Tensor(pad):fill(1)
      elseif self.mirror == 'FRONT_COPY' then
        front_fill_idx = torch.Tensor(pad * 2):fill(1)
      else
        assert(false, 'Unknown mirror option.')
      end
      
      -- rear
      local rear_fill_idx
      if self.mirror == 'FLIP' then
        rear_fill_idx = torch.range(T - 1, math.max(T - pad, 1), -1)
        local fill_idx_len = rear_fill_idx:nElement()
        if fill_idx_len < pad then
          rear_fill_idx = torch.cat({rear_fill_idx, torch.Tensor(pad - fill_idx_len):fill(1)}, 1)
        end
      elseif self.mirror == 'COPY' then
        rear_fill_idx = torch.Tensor(pad):fill(T)
      elseif self.mirror == 'FRONT_COPY' then
        rear_fill_idx = torch.Tensor()
      else
        assert(false, 'Unknown mirror option.')
      end
      
      -- assemble
      if front_fill_idx:nElement() > 0 and rear_fill_idx:nElement() > 0 then
        local front_images = images:index(1, front_fill_idx:long())
        local rear_images = images:index(1, rear_fill_idx:long())
        images = torch.cat({front_images, images, rear_images}, 1)
      elseif front_fill_idx:nElement() > 0 then
        local front_images = images:index(1, front_fill_idx:long())
        images = torch.cat({front_images, images}, 1)
      elseif rear_fill_idx:nElement() > 0 then
        local rear_images = images:index(1, rear_fill_idx:long())
        images = torch.cat({images, rear_images}, 1)
      else
        assert(false, 'Weird thing happened.')
      end
      
      local filled_boxes = {}
      for _, idx in ipairs(front_fill_idx:totable()) do
        table.insert(filled_boxes, boxes[idx]:clone())
      end
      for idx = 1, T do
        table.insert(filled_boxes, boxes[idx])
      end
      for _, idx in ipairs(rear_fill_idx:totable()) do
        table.insert(filled_boxes, boxes[idx]:clone())
      end
      boxes = filled_boxes
    end
    
    collectgarbage()
    local output_coll, bbox_pred_coll, feat = self.DETECTOR(self.detec, 
                                  images, boxes, self.batch_T, self.data_parallel_n, true)
    
    if self.mirror and T >= self.mirror_min_len then
      if self.mirror == 'FLIP' or self.mirror == 'COPY' then
        output_coll = {table.unpack(output_coll, pad + 1, pad + T)}
        bbox_pred_coll = {table.unpack(bbox_pred_coll, pad + 1, pad + T)}
      elseif self.mirror == 'FRONT_COPY' then
        output_coll = {table.unpack(output_coll, 2 * pad + 1, 2* pad + T)}
        bbox_pred_coll = {table.unpack(bbox_pred_coll, 2 * pad + 1, 2* pad + T)}
      else
        assert(false, 'Unknow mirror style.')
      end
    end
    
    -- Clear memory
    collectgarbage()
    
    if not self.glb_map then
      self.glb_map = {name_to_idx = {}, ptr = 1}
    end
    
    for frm_idx = 1, T do
      local frame_idx = frame_seq[frm_idx]
      -- register global index
      local glb_img_id = video_name .. '/' .. self.loader.anno[video_name].im_list[frame_idx]
      if not self.glb_map.name_to_idx[glb_img_id] then
        self.glb_map.name_to_idx[glb_img_id] = self.glb_map.ptr
        self.glb_map.ptr = self.glb_map.ptr + 1
      end
      
      local img_boxes = tds.hash()
      local output, bbox_pred = output_coll[frm_idx], bbox_pred_coll[frm_idx]
      local num_classes = output:size(2) - 1
      local bbox_pred_tmp = bbox_pred:view(-1, 2)
      bbox_pred_tmp:select(2,1):clamp(1, W)
      bbox_pred_tmp:select(2,2):clamp(1, H)
      local nms_timer = torch.Timer()
      for j = 1, num_classes do
        local scores = output:select(2, j+1)
        local idx = torch.range(1, scores:numel()):long()
        local idx2 = scores:gt(thresh[j])
        idx = idx[idx2]
        local scored_boxes = torch.FloatTensor(idx:numel(), 5)
        if scored_boxes:numel() > 0 then
           local bx = scored_boxes:narrow(2, 1, 4)
           bx:copy(bbox_pred:narrow(2, j*4+1, 4):index(1, idx))
           scored_boxes:select(2, 5):copy(scores[idx2])
           scored_boxes = utils.nms(scored_boxes, self.nms_thresh)
        end
        img_boxes[j] = scored_boxes
      end
      self.threads:synchronize()
      local nms_time = nms_timer:time().real
      
      -- write results into text file which will then be fed to eval toolkit
      -- adjust the bbox_pred
      local glb_idx
      if self.loader.anno[video_name].global_idx then
        glb_idx = self.loader.anno[video_name].global_idx[frame_idx]
      else
        glb_idx = self.glb_map.name_to_idx[glb_img_id]
      end
      local gt_hgt = self.loader.anno[video_name].im_size[{frame_idx, 1}]
      local gt_wid = self.loader.anno[video_name].im_size[{frame_idx, 2}]
      local scaler_to_rel = torch.FloatTensor({1.0/W, 1.0/H, 1.0/W, 1.0/H}):view(1, 4)
      local scaler_to_abs = torch.FloatTensor({gt_wid, gt_hgt, gt_wid, gt_hgt}):view(1, 4)
      
      if self.save_MatchTransCoef then
        feat_coll[frame_idx] = feat[{{frm_idx}, {}, {}, {}}]
      end
      
      if self.save_mat_file then
        local save_boxes = bbox_pred:narrow(2, 1, 4)
        save_boxes:cmul(scaler_to_rel:expandAs(save_boxes))
        save_boxes:cmul(scaler_to_abs:expandAs(save_boxes)):round()        
        save_boxes = utils.calibrate_box(save_boxes, gt_hgt, gt_wid)
        local save_scores = output:narrow(2, 2, self.num_classes)
        local max_pos_score = torch.max(save_scores, 2):view(-1)
        local idx = torch.range(1, max_pos_score:numel()):long()
        local idx2 = max_pos_score:gt(thresh[1])
        idx = idx[idx2]
        if idx:nElement() > 0 then
          save_boxes = save_boxes:index(1, idx)
          save_scores = save_scores:index(1, idx)
          -- nms to remove near-duplicates
          if self.remove_near_dup then
            max_pos_score = max_pos_score:index(1, idx)
            local nms_keep_idx = utils.nms_dense(torch.cat({save_boxes, max_pos_score:view(-1, 1)}, 2), 0.95)
            save_boxes = save_boxes:index(1, nms_keep_idx)
            save_scores = save_scores:index(1, nms_keep_idx)
          end
          -- concatenation
          save_boxes = torch.cat({save_boxes, save_scores}, 2)
          local save_filename = paths.concat(paths.dirname(eval_filename), 'mat', video_name, string.format('%d.mat', frame_idx))
          paths.mkdir(paths.dirname(save_filename))
          matio.save(save_filename, save_boxes)
          -- save for the sake of computing offset
          if self.save_MatchTransCoef then
            detailed_boxes_coll[frame_idx] = save_boxes
          end
        end
      end
      
      
      for cls_idx = 1, #img_boxes do
        local cls_img_boxes = img_boxes[cls_idx]
        if cls_img_boxes:nElement() > 0 then
          local img_boxes_only = cls_img_boxes:narrow(2, 1, 4)
          img_boxes_only:cmul(scaler_to_rel:expandAs(img_boxes_only))
        end
        -- evaluate on the fly
        if self.evaluate_on_the_fly then
          self.eval_stat = self:evaluate_GT(cls_img_boxes, cls_idx, video_name, frame_idx, 
                            self.loader.anno, self.eval_stat)
        end        
        if cls_img_boxes:nElement() > 0 then
          local cls_score_boxes = cls_img_boxes:clone()
          local cls_boxes = cls_score_boxes:narrow(2, 1, 4)
          cls_boxes:cmul(scaler_to_abs:expandAs(cls_boxes)):round()        
          cls_boxes = utils.calibrate_box(cls_boxes, gt_hgt, gt_wid)
          if eval_filename ~= nil then
            self:writetxt_box(cls_score_boxes, cls_idx, glb_idx, eval_filename)
          end
        end
      end
      
      table.insert(glb_idx_coll, glb_idx)
      table.insert(img_boxes_coll, img_boxes)
    end
  end
  
  if self.save_MatchTransCoef then
    local win = 2
    local feat_coll = torch.cat(feat_coll, 1)
    local feat_T, feat_H, feat_W = feat_coll:size(1), feat_coll:size(3), feat_coll:size(4)
    for t = 1, feat_T - 1 do
      if detailed_boxes_coll[t] ~= nil then
        local cur_boxes = detailed_boxes_coll[t]:narrow(2, 1, 4)
        local cur_feat = feat_coll[{t, {}, {}, {}}]
        local next_feat = feat_coll[{t+1, {}, {}, {}}]
        local gt_H = self.loader.anno[video_name].im_size[{t, 1}]
        local gt_W = self.loader.anno[video_name].im_size[{t, 2}]
        local shifted_boxes = self:predict_offset(cur_boxes, cur_feat, next_feat, gt_H, gt_W, win)
        -- save
        local save_filename = paths.concat(paths.dirname(eval_filename), 'mat', video_name, string.format('mot_%d.mat', t))
        paths.mkdir(paths.dirname(save_filename))
        matio.save(save_filename, shifted_boxes)
      end
    end
  end
  
  glb_idx_coll = torch.LongTensor(glb_idx_coll)
  return glb_idx_coll, img_boxes_coll
end


function Tester:predict_offset(boxes, cur_feat, next_feat, gt_H, gt_W, win)
  boxes = boxes:clone()
  local feat_H, feat_W = cur_feat:size(2), cur_feat:size(3)
  cur_feat = cur_feat:view(-1, feat_H*feat_W)
  next_feat = next_feat:view(-1, feat_H*feat_W)
  local cur_norm = torch.norm(cur_feat, 2, 1):view(feat_H*feat_W, 1):add(1e-8)
  local next_norm = torch.norm(next_feat, 2, 1):view(1, feat_H*feat_W):add(1e-8)
  local aff = torch.mm(cur_feat:t(), next_feat)
  aff:cdiv(cur_norm:expandAs(aff))
  aff:cdiv(next_norm:expandAs(aff))
  aff = aff:view(feat_H, feat_W, feat_H, feat_W)
  -- predict offset from MatchTrans coef
  local ratio_H = feat_H / gt_H
  local ratio_W = feat_W / gt_W
  local scaler = torch.FloatTensor({ratio_W, ratio_H, ratio_W, ratio_H}):view(1, 4)
  boxes:cmul(scaler:expandAs(boxes)):round()
  boxes = utils.calibrate_box(boxes, feat_H, feat_W)
  boxes = self:predict_offset_driver(boxes, aff, win)
  -- scale back to the image frame
  boxes:cdiv(scaler:expandAs(boxes))
  boxes = utils.calibrate_box(boxes:round(), gt_H, gt_W)
  return boxes
end

function Tester:predict_offset_driver(boxes, aff, win)
  local N = boxes:size(1)
  local feat_H, feat_W = aff:size(3), aff:size(4)
  for idx = 1, N do
    local cur_box = boxes:select(1, idx)
    local offset = {0, 0} -- {x, y}
    local box_pixcount = 0
    for x1 = cur_box[1], cur_box[3] do
      for y1 = cur_box[2], cur_box[4] do
        -- inner loop
        local shifted_ptr = {0, 0} -- {x, y}
        local total_weight = 0
        for x2 = x1-win, x1+win do
          for y2 = y1-win, y1+win do
            if x2 <= feat_W and x2 >= 1 and y2 <= feat_H and y2 >= 1 then
              local weight = aff[{y1, x1, y2, x2}]
              shifted_ptr[1] = shifted_ptr[1] + weight * x2
              shifted_ptr[2] = shifted_ptr[2] + weight * y2
              total_weight = total_weight + weight
            end
          end
        end
        shifted_ptr[1] = shifted_ptr[1] / total_weight
        shifted_ptr[2] = shifted_ptr[2] / total_weight
        -- compute the offset for {x1, y1}
        offset[1] = offset[1] + shifted_ptr[1] - x1
        offset[2] = offset[2] + shifted_ptr[2] - y1
        box_pixcount = box_pixcount + 1
      end
    end
    -- compute an average offset for all pixs in the box
    offset[1] = offset[1] / box_pixcount
    offset[2] = offset[2] / box_pixcount
    cur_box[1] = cur_box[1] + offset[1]
    cur_box[3] = cur_box[3] + offset[1]
    cur_box[2] = cur_box[2] + offset[2]
    cur_box[4] = cur_box[4] + offset[2]
  end
  return boxes
end


function Tester:evalOne(eval_filename, video_name, start_frm)
  local thresh = self.thresh
  
  collectgarbage()
  
  local images, boxes, record
  if video_name and start_frm then
    images, boxes, record = self.loader:light_sample(nil, video_name, start_frm)
  else
    images, boxes, record = self.loader:light_sample(nil)
  end
  
  local glb_idx_coll, img_boxes_coll, mem_coll, conv_coll, z_coll, r_coll = {}, {}, {}, {}, {}, {}
  for vid_idx = 1, self.loader.batch_N do
    local vid_images = images[vid_idx]
    local vid_boxes = boxes[vid_idx]
    local H, W = vid_images:size(3), vid_images:size(4)    
    
    local output_coll, bbox_pred_coll
    if #self.detec.model:findModules('nn.STMM') > 0 then
      output_coll, bbox_pred_coll = self.detec:detect_VID_NO_OVERLAP(
                              vid_images, vid_boxes, self.batch_T, self.data_parallel_n, true)
    else
      output_coll, bbox_pred_coll = self.detec:detect_VID_SEQ(
                              vid_images, vid_boxes, self.batch_T, self.data_parallel_n, true)
    end
    
    collectgarbage()
    if self.inference_method == 'static' then
      -- Inference type I: per-image inference
      for frm_idx = 1, self.loader.batch_T do
        local img_boxes = tds.hash()
        local output, bbox_pred = output_coll[frm_idx], bbox_pred_coll[frm_idx]
        local num_classes = output:size(2) - 1
        local bbox_pred_tmp = bbox_pred:view(-1, 2)
        bbox_pred_tmp:select(2,1):clamp(1, W)
        bbox_pred_tmp:select(2,2):clamp(1, H)
        local nms_timer = torch.Timer()
        for j = 1, num_classes do
          local scores = output:select(2, j+1)
          local idx = torch.range(1, scores:numel()):long()
          local idx2 = scores:gt(thresh[j])
          idx = idx[idx2]
          local scored_boxes = torch.FloatTensor(idx:numel(), 5)
          if scored_boxes:numel() > 0 then
             local bx = scored_boxes:narrow(2, 1, 4)
             bx:copy(bbox_pred:narrow(2, j*4+1, 4):index(1, idx))
             scored_boxes:select(2, 5):copy(scores[idx2])
             scored_boxes = utils.nms(scored_boxes, self.nms_thresh)
          end
          img_boxes[j] = scored_boxes
        end
        self.threads:synchronize()
        local nms_time = nms_timer:time().real
        
        -- write results into text file which will then be fed to eval toolkit
        -- adjust the bbox_pred
        local video_name = record.vid[vid_idx]
        local frame_idx = record.frm[video_name][frm_idx]
        local glb_idx
        if self.loader.anno[video_name].global_idx then
          glb_idx = self.loader.anno[video_name].global_idx[frame_idx]
        else
          glb_idx = -1
        end
        local gt_hgt = self.loader.anno[video_name].im_size[{frame_idx, 1}]
        local gt_wid = self.loader.anno[video_name].im_size[{frame_idx, 2}]
        local scaler_to_rel = torch.FloatTensor({1.0/W, 1.0/H, 1.0/W, 1.0/H}):view(1, 4)
        local scaler_to_abs = torch.FloatTensor({gt_wid, gt_hgt, gt_wid, gt_hgt}):view(1, 4)
        for cls_idx = 1, #img_boxes do
          local cls_img_boxes = img_boxes[cls_idx]
          if cls_img_boxes:nElement() > 0 then
            local img_boxes_only = cls_img_boxes:narrow(2, 1, 4)
            img_boxes_only:cmul(scaler_to_rel:expandAs(img_boxes_only))
          end
          -- evaluate on the fly
          if self.evaluate_on_the_fly then
            self.eval_stat = self:evaluate_GT(cls_img_boxes, cls_idx, video_name, frame_idx, 
                                self.loader.anno, self.eval_stat)
          end
          if cls_img_boxes:nElement() > 0 then
            -- generate box for file writing
            local cls_score_boxes = cls_img_boxes:clone()
            local cls_boxes = cls_score_boxes:narrow(2, 1, 4)
            cls_boxes:cmul(scaler_to_abs:expandAs(cls_boxes)):round()        
            cls_boxes = utils.calibrate_box(cls_boxes, gt_hgt, gt_wid)
            if eval_filename ~= nil then
              self:writetxt_box(cls_score_boxes, cls_idx, glb_idx, eval_filename)
            end
          end
        end
        
        table.insert(glb_idx_coll, glb_idx)
        table.insert(img_boxes_coll, img_boxes)
      end
    else
      assert(false, 'Unknown inference method.')
    end
  end
  
  glb_idx_coll = torch.LongTensor(glb_idx_coll)
  return glb_idx_coll, img_boxes_coll, record, mem_coll, conv_coll, z_coll, r_coll, boxes
end


function Tester:evaluate_GT(pred_bboxes, cls_idx, video_name, frame_idx, anno, eval_stat)
  local objs = anno[video_name].obj
  -- init eval_stat
  if not self.gt_record then
    self.gt_record = {}
  end
  if not eval_stat then
    eval_stat = {}
    for cat_name, _ in pairs(self.loader.cat_name_to_id) do
      eval_stat[cat_name] = {check_map={}, total_obj=0}
    end
  end
  local pred_category = self.loader.cat_id_to_name[cls_idx + 1]  -- since cls_idx starts from 1
  local check_map, check_col
  if pred_bboxes:nElement() > 0 then
    local y, i = torch.sort(pred_bboxes:select(2, 5), 1, true)
    pred_bboxes = pred_bboxes:index(1, i)
    check_map = torch.FloatTensor(pred_bboxes:size(1), 2):zero()
    check_map:select(2, 2):copy(pred_bboxes:select(2, 5))
    check_col = check_map:select(2, 1)
  end
  for obj_idx, obj in ipairs(objs) do
    if pred_category == obj.category then
      if pred_bboxes:nElement() > 0 then
        -- decide if this GT has been detected before
        local detected = false
        pcall(function () detected = self.gt_record[video_name][obj_idx][frame_idx] end)
        --assert(not detected, 'Duplicated detection on same frame.')
        if obj.start_frame <= frame_idx and obj.end_frame >= frame_idx and not detected then
          local gt_box = obj.boxes[{frame_idx - obj.start_frame + 1, {3, 6}}]
          local overlap = utils.boxoverlap_01(pred_bboxes[{{}, {1, 4}}], gt_box)
          overlap = overlap:ge(self.eval_iou_thresh)
          overlap = overlap:float()
          local overlap_nonzero_idx = overlap:nonzero()
          if overlap_nonzero_idx:nElement() > 0 then
            overlap_nonzero_idx = torch.min(overlap:nonzero())
            overlap:zero()
            overlap[overlap_nonzero_idx] = 1
          end
          if torch.sum(overlap) > 0 then
            if not self.gt_record[video_name] then
              self.gt_record[video_name] = {}
            end
            if not self.gt_record[video_name][obj_idx] then
              self.gt_record[video_name][obj_idx] = {}
            end
            self.gt_record[video_name][obj_idx][frame_idx] = true
          end
          check_col:cmax(overlap)
        end
      end
    end
  end
  -- put back the evaluated results
  if check_map then
    table.insert(eval_stat[pred_category].check_map, check_map)
  end
  return eval_stat
end





