
local myutils = require 'myutils'
local utils = require 'utils'
local image = require 'image'
local BatchProviderVID = torch.class('fbcoco.BatchProviderVID')

-- DEBUG flag
local DEBUG = false

function BatchProviderVID:__init(anno, transformer, opt)
   assert(transformer,'must provide transformer!')
   self.anno = anno
   self.prop_dir = opt.prop_dir
   self.fg_threshold = opt.fg_threshold
   self.bg_threshold = opt.bg_threshold
   self.fg_fraction = opt.fg_fraction
   self.batch_size = opt.batch_size
   self.test_batch_size = opt.test_batch_size
   self.sample_n_per_box = opt.sample_n_per_box
   self.sample_sigma = opt.sample_sigma
   self.batch_N = opt.seq_per_batch or 2
   self.batch_T = opt.timestep_per_batch or 16
   self.frame_stride = opt.frame_stride
   self.spec_im_size = opt.spec_im_size
   self.scale = opt.scale or 600
   self.max_size = opt.max_size or 1000
   self.image_transformer = transformer
   self.img_dir = opt.img_dir
   -- how many threads for Parallel feeding
   self.parallel_roi_batch = opt.parallel_roi_batch or 1
   -- whether to focus on the center or not
   self.seq_center = opt.seq_center or false
   -- uniformly jitter the scale by this frac
   self.scale_jitter = opt.scale_jitter or 0 -- default to 0    
   -- uniformly jitter the scale by this frac
   self.aspect_jitter = opt.aspect_jitter or 0 -- default to 0
   -- likelihood of doing a random crop (in each dimension, independently)
   self.crop_likelihood = crop_likelihood or 0 
   -- number of attempts to try to find a valid crop
   self.crop_attempts = 10                     
   -- a crop must preserve at least this fraction of the iamge
   self.crop_min_frac = 0.7
   -- get a deterministic list of video names
   self.video_names = myutils.keys(anno)
   self.non_redundant_sampling = opt.sampling_mode == 'NONE_REDUNDANT' -- 'RANDOM' or 'NONE_REDUNDANT'
   -- init the video reading list
   self:shuffle()
   if self.non_redundant_sampling then
    self:init_sampling_hist()
   end
   
   -- augment the category name list
   local category_list
   if opt.dataset == 'ImageNetVID' then
    category_list = {'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 
                'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 
                'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 
                'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra'
     }
   else
    assert(false, 'Unknown dataset.')
   end
   self.cat_name_to_id = {}
   self.cat_id_to_name = {}
   self.class_num = #category_list
   for cat_id, cat_name in ipairs(category_list) do
     self.cat_name_to_id[cat_name] = cat_id + 1 -- +1 because we leave 1 to background class
     self.cat_id_to_name[cat_id + 1] = cat_name
   end
   
   -- initialize a transformer
   if opt.brightness_var and opt.contrast_var and opt.saturation_var and opt.lighting_var then
     self.color_jitter = fbcoco.ColorTransformer(opt.brightness_var, 
                          opt.contrast_var, opt.saturation_var, opt.lighting_var)
   else
     self.color_jitter = nil
   end
    
   -- init a counter array recording the encounter of objects of different categories
   self.cat_counter = torch.FloatTensor(self.class_num + 1):zero()   
end

function BatchProviderVID:load_prop(video_name)
  local filename = paths.concat(self.prop_dir, string.format('%s.t7', video_name))
  local res = torch.load(filename)
  return res
end

function BatchProviderVID:init_sampling_hist()
  self.sampling_hist = {}
  for vid_name, vid in pairs(self.anno) do
    self.sampling_hist[vid_name] = torch.IntTensor(#vid.im_list):zero()
  end
end

function BatchProviderVID:get_next_video_idx()
  self.video_ptr = self.video_ptr + 1
  if self.video_ptr > self.video_index_list:nElement() then
    self:shuffle()
  end
  local video_idx = self.video_index_list[self.video_ptr]
  return video_idx
end

function BatchProviderVID:shuffle()
  self.video_index_list = torch.randperm(#self.video_names):long()
  self.video_ptr = 1
end

function BatchProviderVID:getFlows(video_name, frame_seq, flip, transform, spec_im_size)
  -- sample some jittering parameters
  flip = flip == 1
  -- transform flag
  if transform == nil then
    transform = true
  else
    transform = transform == 1
  end
  local num_images = frame_seq:nElement()
  local im_s, im_scale
  
  -- load images from disk
  local flows
  for ii, frame_idx in ipairs(frame_seq:totable()) do
    local concise_video_name = paths.basename(video_name)
    local ptr = frame_idx
    local u_filename = paths.concat(self.img_dir, 'u', concise_video_name, string.format('frame%.6d.jpg', ptr))  
    while not myutils.file_exists(u_filename) do
      --assert(frame_idx - ptr <= 3, 'Fallback too much.')
      if frame_idx - ptr > 3 then
        print(string.format('WARNING: file -- %s, |target_idx - actual_idx| = %d', u_filename, frame_idx - ptr))
      end
      ptr = ptr - 1
      u_filename = paths.concat(self.img_dir, 'u', concise_video_name, string.format('frame%.6d.jpg', ptr)) 
    end
    local u = image.load(u_filename, 1, 'float')
    local ptr = frame_idx
    local v_filename = paths.concat(self.img_dir, 'v', concise_video_name, string.format('frame%.6d.jpg', ptr))    
    while not myutils.file_exists(v_filename) do
      --assert(frame_idx - ptr <= 3, 'Fallback too much.')
      if frame_idx - ptr > 3 then
        print(string.format('WARNING: file -- %s, |target_idx - actual_idx| = %d', v_filename, frame_idx - ptr))
      end
      ptr = ptr - 1
      v_filename = paths.concat(self.img_dir, 'v', concise_video_name, string.format('frame%.6d.jpg', ptr))  
    end
    local v = image.load(v_filename, 1, 'float')
    if ii == 1 then
      if spec_im_size ~= nil then
        im_s = spec_im_size
      else
        local im_size = u[1]:size()
        local im_size_min = math.min(im_size[1],im_size[2])
        local im_size_max = math.max(im_size[1],im_size[2])
        im_scale = self.scale/im_size_min
        im_scale = {im_scale, im_scale}
        im_s = {math.ceil(im_size[1]*im_scale[1]), math.ceil(im_size[2]*im_scale[1])}
        for dim = 1,2 do
          if im_s[dim] > self.max_size then
            local rat = im_s[dim] / self.max_size
            im_s = {math.ceil(im_s[1] / rat), math.ceil(im_s[2] / rat)}
            im_scale = {im_scale[1] / rat, im_scale[2] / rat}
          end
        end
      end
      im_s = {math.floor(im_s[1]), math.floor(im_s[2])}
      flows = torch.FloatTensor(num_images, 2, im_s[1], im_s[2]):zero()
    end
    if flip then
      u = image.hflip(u)
      u = 1 - u
      v = image.hflip(v)
    end
    local flow = torch.cat({u, v}, 1)
    flow = self:scale_flow(flow, im_s[1], im_s[2])
    if transform then
      flow = self.image_transformer:forward(flow)
    end
    flows[ii]:copy(flow)
  end
  im_s = torch.FloatTensor(im_s)
  
  return flows, im_s
end

function BatchProviderVID:scale_flow(flow, hgt, wid)
  -- flow: [2, H, W], first x (horizontal) then y
  local H, W = flow:size(2), flow:size(3)
  local u = image.scale(flow[1], wid, hgt)
  local v = image.scale(flow[2], wid, hgt)
  --u:mul(wid/W)
  --v:mul(hgt/H)
  local scaled_flow = torch.cat({u:view(1, hgt, wid), v:view(1, hgt, wid)}, 1)
  return scaled_flow
end

function BatchProviderVID:getImages(video_name, frame_seq, flip, transform, spec_im_size)
  -- Load one video chunk 
  local imgs = {}
  
  -- transform flag
  if transform == nil then
    transform = true
  else
    transform = transform == 1
  end
  
  -- sample some jittering parameters
  local num_images = frame_seq:nElement()
  local aspect_jitter = 1 + (torch.uniform(-1.0,1.0))*self.aspect_jitter
  local scale_jitter  = 1 + (torch.uniform(-1.0,0.0))*self.scale_jitter
  --local aspect_jitter = 1 + (torch.uniform(-1.5,0.5))*self.aspect_jitter
  --local scale_jitter  = 1 + (torch.uniform(-1.5,0.5))*self.scale_jitter
  
  flip = flip == 1
  local im_s, im_scale, expand_shape
  
  if self.color_jitter then
    self.color_jitter:SampleColorParams()
  end
  
  -- load images from disk
  local im_set = {}
  for ii, frame_idx in ipairs(frame_seq:totable()) do
    local img_filename = self.anno[video_name].im_list[frame_idx]
    img_filename = paths.concat(self.img_dir, video_name, img_filename)
    local im = image.load(img_filename, 3, 'double')
    im_set[ii] = im
  end
  
  for ii, frame_idx in ipairs(frame_seq:totable()) do
    local im = im_set[ii]
    -- perform photometric transformation
    if self.color_jitter then
      im = self.color_jitter:forward(im)
    end
    
    if transform then
      im = self.image_transformer:forward(im)
    end
    if flip then im = image.hflip(im) end
    
    if ii == 1 then
      if spec_im_size ~= nil then
        im_s = spec_im_size
      else
        local im_size = im[1]:size()
        local im_size_min = math.min(im_size[1],im_size[2])
        local im_size_max = math.max(im_size[1],im_size[2])
        im_scale = self.scale/im_size_min
        im_scale = im_scale * scale_jitter
        im_scale = {im_scale * math.sqrt(aspect_jitter), im_scale / math.sqrt(aspect_jitter)}
        im_s = {math.ceil(im_size[1]*im_scale[1]), math.ceil(im_size[2]*im_scale[2])}
        for dim = 1,2 do
          if im_s[dim] > self.max_size then
            local rat = im_s[dim] / self.max_size
            im_s = {math.ceil(im_s[1] / rat), math.ceil(im_s[2] / rat)}
            im_scale = {im_scale[1] / rat, im_scale[2] / rat}
          end
        end
      end
      im_s = {math.floor(im_s[1]), math.floor(im_s[2])}
      expand_shape = torch.LongStorage({1, 3, im_s[1], im_s[2]})
    end
    table.insert(imgs, image.scale(im,im_s[2],im_s[1]):view(expand_shape))
  end
  
  imgs = torch.cat(imgs, 1)
  im_s = torch.FloatTensor(im_s)
  return imgs, im_s
end

function BatchProviderVID:sampleAroundGTBoxes(boxes, n_per_box, sigma)
   local samples = torch.repeatTensor(boxes, n_per_box, 1)
   return samples:add(torch.FloatTensor(#samples):normal(0,sigma))
end

function BatchProviderVID:organize_boxes(gtboxes, rois, gtlabels, obj_n)
  if self.sample_n_per_box > 0 and gtboxes:numel() > 0 then
    local sampled = self:sampleAroundGTBoxes(gtboxes:narrow(2, 1, 4), self.sample_n_per_box, self.sample_sigma)
    rois = rois:cat(sampled, 1)
  end
  
  -- compute IOU
  local overlap = torch.FloatTensor(gtboxes:size(1), rois:size(1))
  for ii = 1, gtboxes:size(1) do
    local gtbox = gtboxes[ii]
    local o = utils.boxoverlap_01(rois, gtbox)
    overlap[ii] = o
  end
  local max_overlap, max_idx = torch.max(overlap, 1)
  max_overlap = max_overlap:view(-1)
  max_idx = max_idx:view(-1)
  local labels = gtlabels:index(1, max_idx)
  
  -- pick the foreground and background boxes
  local fg = max_overlap:ge(self.fg_threshold):nonzero()  
  local fg_boxes, fg_labels, fg_max_idx = nil, nil, nil
  if fg:nElement() > 0 then
    fg = fg:view(-1)
    local fg_num = fg:nElement()
    fg = fg:index(1, torch.randperm(fg_num)[{{1, math.min(fg_num, self.fg_num_each)}}]:long())
    fg_boxes = rois:index(1, fg)
    fg_labels = labels:index(1, fg)
    fg_max_idx = max_idx:index(1, fg)
  end
  local bg = max_overlap:ge(self.bg_threshold[1]):cmul(max_overlap:lt(
              self.bg_threshold[2])):nonzero()
  local bg_boxes, bg_labels, bg_max_idx = nil, nil, nil
  if bg:nElement() > 0 then
    bg = bg:view(-1)
    local bg_num = bg:nElement()
    bg = bg:index(1, torch.randperm(bg_num)[{{1, math.min(bg_num, self.bg_num_each)}}]:long())
    bg_boxes = rois:index(1, bg)
    bg_labels = torch.IntTensor(bg:nElement()):fill(1)
    bg_max_idx = max_idx.new(bg:nElement()):zero()
  end
  
  local boxes, labels, correspondance = nil, nil, nil
  local obj_id = gtboxes[{{}, 5}]:clone()
  local obj_box = torch.FloatTensor(obj_n, 4):fill(-1)
  for idx = 1, gtboxes:size(1) do
    local cur_obj_id = obj_id[idx]
    obj_box[{cur_obj_id, {}}]:copy(gtboxes[{idx, {1, 4}}])
  end
  gtboxes = gtboxes[{{}, {1, 4}}]
  
  if bg:nElement() > 0 and fg:nElement() > 0 then
    boxes = torch.cat({fg_boxes, bg_boxes}, 1)
    labels = torch.cat({fg_labels, bg_labels}, 1)
    correspondance = torch.cat({fg_max_idx, bg_max_idx}, 1)
  else 
    if bg:nElement() > 0 then
      boxes = bg_boxes
      labels = bg_labels
      correspondance = bg_max_idx
    elseif fg:nElement() > 0 then
      boxes = fg_boxes
      labels = fg_labels
      correspondance = fg_max_idx
    else
      boxes = torch.FloatTensor(0, 4)
      labels = torch.IntTensor(0)
      correspondance = torch.LongTensor(0)
    end
  end
  
  -- add the ground truth box
  if boxes:nElement() > 0 then
    boxes = torch.cat({gtboxes, boxes}, 1)
    labels = torch.cat({gtlabels, labels}, 1)
    correspondance = torch.cat({torch.range(1, gtlabels:nElement()):type(
                      correspondance:type()), correspondance}, 1)
  else
    boxes = gtboxes
    labels = gtlabels
    correspondance = torch.range(1, gtlabels:nElement()):type(correspondance:type())
  end
  boxes = boxes:contiguous()
  
  return {boxes = boxes, labels = labels, correspondance = correspondance, 
            obj_id = obj_id, obj_box = obj_box}
end

function BatchProviderVID:gen_rois(box_coll, im_size_coll)
  local img_counter = 1
  local rois, labels, gtboxes = {}, {}, {}
  for vid_idx, vid in ipairs(box_coll) do
    local im_size = im_size_coll[vid_idx]
    for fr_idx, fr in ipairs(vid) do
      local fr_labels = fr.labels
      local fr_correspondance = fr.correspondance
      local fr_boxes = fr.boxes
      local fr_obj_id = fr.obj_id
      fr_boxes[{{}, 1}]:mul(im_size[2])
      fr_boxes[{{}, 3}]:mul(im_size[2])
      fr_boxes[{{}, 2}]:mul(im_size[1])
      fr_boxes[{{}, 4}]:mul(im_size[1])
      fr_boxes = utils.calibrate_box(torch.round(fr_boxes), im_size[1], im_size[2])
      
      local fr_gtboxes = fr_boxes:clone():zero()
      local fg_idx = fr_labels:ge(2):nonzero():view(-1) -- all labels > 1 (>= 2)
      local fg_correspondance = fr_correspondance:index(1, fg_idx)
      --fr_gtboxes:index(1, fg_idx):copy(fr_boxes:index(1, fg_correspondance))
      fr_gtboxes:indexCopy(1, fg_idx, fr_boxes:index(1, fg_correspondance))
      fr_boxes = torch.cat({fr_boxes.new(fr_boxes:size(1), 1):fill(img_counter), fr_boxes}, 2)
      
      -- this is not the last frame
      if fr_idx ~= #vid then
        
        -- Below are two ways of computing the motion prediction target:
        --  1) Making use of all data, from any box --> gtbox at next frame
        --  2) Only do prediction from gtbox at current frame --> gtbox at next frame
        --local obj_idx = fr_obj_id:index(1, fg_correspondance):long()
        --local tmp_idx = fg_idx
        local obj_idx = fr_obj_id:long()
        local tmp_idx = torch.range(1, obj_idx:nElement()):long()
        
        local nxt_fr = vid[fr_idx+1]
        local nxt_fr_obj_box = nxt_fr.obj_box        
        local cand_nxt_fr_gtboxes = nxt_fr_obj_box:index(1, obj_idx)
        local valid = cand_nxt_fr_gtboxes[{{}, 1}]:ge(0):nonzero():view(-1)
        cand_nxt_fr_gtboxes = cand_nxt_fr_gtboxes:index(1, valid)
        tmp_idx = tmp_idx:index(1, valid)
      end
      
      table.insert(rois, fr_boxes)
      table.insert(gtboxes, fr_gtboxes)
      table.insert(labels, fr_labels)
      img_counter = img_counter + 1
    end
  end
  
  -- flatten
  rois = torch.cat(rois, 1)
  gtboxes = torch.cat(gtboxes, 1)
  labels = torch.cat(labels, 1)
  
  -- compute regression target
  local bboxregr_vals = torch.FloatTensor(rois:size(1), 4*(self.class_num + 1)):zero()
  for i,label in ipairs(labels:totable()) do
     if label > 1 then
        local out = bboxregr_vals[i]:narrow(1,(label-1)*4 + 1,4)
        utils.convertTo(out, rois[i]:narrow(1,2,4), gtboxes[i])
        out:add(-1,self.bbox_regr.mean):cdiv(self.bbox_regr.std)
     end
  end

  return rois, labels, bboxregr_vals
end


function BatchProviderVID:squeeze_im_to_tensor(images, im_size)
  -- create single tensor with all images, padding with zero for different sizes
  im_size = torch.cat(im_size, 2)
  local channel = images[1]:size(2)
  local max_shape = im_size:max(2):view(-1)
  local im_tensor = torch.FloatTensor(self.batch_N,self.batch_T,channel,max_shape[1],max_shape[2]):zero()
  for i,v in ipairs(images) do
    im_tensor[{i, {}, {}, {1,v:size(3)}, {1,v:size(4)}}]:copy(v)
  end
  im_tensor = im_tensor:view(self.batch_N*self.batch_T,channel,max_shape[1],max_shape[2])
  return im_tensor
end

function BatchProviderVID:crop_squeeze_im_to_tensor(images, boxes, im_size)
  -- compute aspect ratios
  local ar = {}
  for idx, siz in ipairs(im_size) do
    ar[idx] = siz[1] / siz[2]
  end
  
  -- create single tensor with all images, padding with zero for different sizes
  im_size = torch.cat(im_size, 2)
  local max_shape = im_size:max(2):view(-1)
  local im_tensor = torch.FloatTensor(self.batch_N,self.batch_T,3,max_shape[1],max_shape[2]):zero()
  for i,v in ipairs(images) do
    im_tensor[{i, {}, {}, {1,v:size(3)}, {1,v:size(4)}}]:copy(v)
  end
  im_tensor = im_tensor:view(self.batch_N*self.batch_T,3,max_shape[1],max_shape[2])
  return im_tensor
end

--------------------------------------------

function BatchProviderVID:light_sample_target(video_name, frame_idx_seq, flat_flag)
  -- perform light sample with certain target, without touching ground-truth
  collectgarbage()
  
  -- loop over videos
  local prop = self:load_prop(video_name)
  local box_coll = {}
  
  for _, frame_idx in ipairs(frame_idx_seq:totable()) do
    -- get proposals
    local rois, roi_scores = self:slice_prop(prop, frame_idx)
    table.insert(box_coll, {rois=rois, roi_scores=roi_scores})
  end
  
  -- get image
  local images, im_sizes = self:getImages(video_name, frame_idx_seq, 0, 0)
  
  -- scale the boxes into absolute coordinate system
  local rois = {}
  local hgt = im_sizes[1]
  local wid = im_sizes[2]
  local scaler = torch.FloatTensor({wid, hgt, wid, hgt}):view(1, 4)
  for frm_idx, frm in ipairs(box_coll) do
    local roi_scores = frm.roi_scores
    rois[frm_idx] = utils.calibrate_box(torch.cmul(
            frm.rois, scaler:expandAs(frm.rois)):round(), hgt, wid)
    if self.test_batch_size then
      local sortval, sortidx = torch.sort(roi_scores, 1, true)
      sortidx = sortidx[{{1, math.min(self.test_batch_size, roi_scores:nElement())}}]
      rois[frm_idx] = rois[frm_idx]:index(1, sortidx)
    end
  end
  return images, rois
end

--------------------------------------------

function BatchProviderVID:light_sample(flat_flag, spec_video_name, spec_start_frm)
  collectgarbage()
  local collected_N = 0
  local box_coll, image_coll, im_size_coll = {}, {}, {}
  local record = {vid={}, frm={}}
  
  -- loop over videos
  while collected_N < self.batch_N do
    
    local video_name
    if not spec_video_name then
      local video_idx = self:get_next_video_idx()
      video_name = self.video_names[video_idx]
    else
      video_name = spec_video_name
    end
    
    local obj = self.anno[video_name].obj
    local prop = self:load_prop(video_name)
    local T = #self.anno[video_name].im_list
    local indicator = torch.ByteTensor(T, #obj):zero()
    local frlen, start_idx, end_idx = {}, {}, {}
    local vid_box_coll = {}
    local gtlabels = {}
    
    -- figure out how many frames are there for each object
    for oi, cur_obj in ipairs(obj) do
      indicator[{{cur_obj.start_frame, cur_obj.end_frame}, oi}]:fill(1)
      frlen[oi] = cur_obj.end_frame - cur_obj.start_frame + 1
      start_idx[oi] = cur_obj.start_frame
      end_idx[oi] = cur_obj.end_frame
      gtlabels[oi] = self.cat_name_to_id[cur_obj.category]
      assert(gtlabels[oi] ~= nil)
    end
    frlen = torch.IntTensor(frlen)
    gtlabels = torch.IntTensor(gtlabels)
    local min_len = (self.batch_T - 1) * self.frame_stride + 1
    local valid_oi = torch.nonzero(frlen:ge(min_len))
    
    if valid_oi:nElement() > 0 then
      valid_oi = valid_oi:view(-1)
      collected_N = collected_N + 1
      record.vid[collected_N] = video_name    
      record.frm[video_name] = {}  
      local oi = valid_oi[{torch.random(valid_oi:nElement())}]
      local oi_start_idx = start_idx[oi]
      local oi_end_idx = end_idx[oi]
      
      local fi_start
      if not self.non_redundant_sampling then
        fi_start = torch.random(oi_start_idx, oi_end_idx - min_len + 1)
      else      
        local hit = self.sampling_hist[video_name][{{oi_start_idx, oi_end_idx - min_len + 1}}]        
        local randperb = torch.randperm(hit:nElement()):type('torch.LongTensor')
        local randperb_hit = hit:index(1, randperb)
        local _, min_idx = torch.min(randperb_hit, 1)
        min_idx = randperb[min_idx[1]]
        hit[min_idx] = hit[min_idx] + 1
        fi_start = oi_start_idx + min_idx - 1
      end
      
      if spec_start_frm then
        fi_start = spec_start_frm
      end
      
      local fi_end = fi_start + min_len - 1
      for _, frame_idx in ipairs(torch.range(fi_start, fi_end, self.frame_stride):totable()) do
        table.insert(record.frm[video_name], frame_idx)
        local frame_valid_oi = torch.nonzero(indicator[frame_idx]):view(-1)
        
        -- get ground truth boxes
        local gtboxes = torch.FloatTensor(frame_valid_oi:nElement(), 4):zero()
        for ii, oii in ipairs(frame_valid_oi:totable()) do
          --local tmp_idx = torch.nonzero(obj[oii].boxes[{{}, 2}]:eq(frame_idx)):view(-1)
          local tmp_idx = frame_idx - obj[oii].start_frame + 1
          gtboxes[ii] = obj[oii].boxes[{tmp_idx, {3, 6}}]
        end
        
        -- get proposals
        local rois, roi_scores = self:slice_prop(prop, frame_idx)
        
        -- select top K
        if self.test_batch_size then
          local sortval, sortidx = torch.sort(roi_scores, 1, true)
          sortidx = sortidx[{{1, math.min(self.test_batch_size, sortidx:nElement())}}]
          rois = rois:index(1, sortidx)
        end
        
        -- match rois and gtboxes
        table.insert(vid_box_coll, {rois=rois, roi_scores=roi_scores, gtboxes=gtboxes})
      end
      table.insert(box_coll, vid_box_coll)
      
      -- get image
      local frame_seq = torch.range(fi_start, fi_end, self.frame_stride)
      local images, im_sizes
      images, im_sizes = self:getImages(video_name, frame_seq, 0, 0)
      table.insert(image_coll, images)
      table.insert(im_size_coll, im_sizes)
      record.frm[video_name] = torch.IntTensor(record.frm[video_name])
    end
  end
  
  -- scale the boxes into absolute coordinate system
  local rois = {}
  for vid_idx, vid in ipairs(box_coll) do
    local hgt = im_size_coll[vid_idx][1]
    local wid = im_size_coll[vid_idx][2]
    local scaler = torch.FloatTensor({wid, hgt, wid, hgt}):view(1, 4)
    local vid_rois = {}
    for frm_idx, frm in ipairs(vid) do
      table.insert(vid_rois, utils.calibrate_box(torch.cmul(
              frm.rois, scaler:expandAs(frm.rois)):round(), hgt, wid))
    end
    table.insert(rois, vid_rois)
  end
  
  if flat_flag then
    image_coll = self:squeeze_im_to_tensor(image_coll, im_size_coll)
    local new_rois = {}
    for vid_idx, vid in ipairs(rois) do
      for frm_idx, frm in ipairs(vid) do
        table.insert(new_rois, frm)
      end
    end
    rois = new_rois
  end
  
  -- pack im_size_coll into record
  record.im_size = im_size_coll  
  
  return image_coll, rois, record
end

function BatchProviderVID:slice_prop(prop, frame_idx)
  local rois, roi_scores
  if torch.isTensor(prop.boxes) then
    rois = prop.boxes[{frame_idx, {}, {}}]
    roi_scores = prop.scores[{{}, frame_idx}]
  elseif torch.type(prop.boxes) == 'table' then
    rois = prop.boxes[frame_idx]
    roi_scores = prop.scores[frame_idx]
  else
    assert(false, 'Unknown proposal format.')
  end
  return rois, roi_scores
end


function BatchProviderVID:sample()
  collectgarbage()
  self.fg_num_each = self.fg_fraction * self.batch_size
  self.bg_num_each = self.batch_size - self.fg_num_each
  local collected_N = 0
  local box_coll, image_coll, im_size_coll = {}, {}, {}
  local do_flip = torch.FloatTensor(self.batch_N):random(0,1)
  local batch_large_hw_ratio
  
  -- loop over videos
  while collected_N < self.batch_N do
    local video_idx = self:get_next_video_idx()
    local video_name = self.video_names[video_idx]
    local obj = self.anno[video_name].obj
    local prop = self:load_prop(video_name)
    local T = #self.anno[video_name].im_list
    local indicator = torch.ByteTensor(T, #obj):zero()
    local frlen, start_idx, end_idx = {}, {}, {}
    local vid_box_coll = {}
    local gtlabels = {}
    
    -- figure out how many frames are there for each object
    for oi, cur_obj in ipairs(obj) do
      indicator[{{cur_obj.start_frame, cur_obj.end_frame}, oi}]:fill(1)
      frlen[oi] = cur_obj.end_frame - cur_obj.start_frame + 1
      start_idx[oi] = cur_obj.start_frame
      end_idx[oi] = cur_obj.end_frame
      gtlabels[oi] = self.cat_name_to_id[cur_obj.category]
      assert(gtlabels[oi] ~= nil)
    end
    frlen = torch.IntTensor(frlen)
    gtlabels = torch.IntTensor(gtlabels)
    local min_len = (self.batch_T - 1) * self.frame_stride + 1
    local valid_oi = torch.nonzero(frlen:ge(min_len))
    
    -- figure out the aspect ratio
    local large_hw_ratio
    if self.anno[video_name].im_size and self.anno[video_name].im_size:nElement() > 0 then
      local hgt = self.anno[video_name].im_size[1][1]
      local wid = self.anno[video_name].im_size[1][2]
      large_hw_ratio = (hgt/wid) > 1
    else
      local img_filename = self.anno[video_name].im_list[1]
      img_filename = paths.concat(self.img_dir, video_name, img_filename)
      local im = image.load(img_filename, 3, 'double')
      large_hw_ratio = (im:size(2) / im:size(3)) > 1
    end 
    if collected_N == 0 then
      batch_large_hw_ratio = large_hw_ratio
    end
    
    
    if valid_oi:nElement() > 0 and batch_large_hw_ratio == large_hw_ratio then
      valid_oi = valid_oi:view(-1)
      collected_N = collected_N + 1
      local flip = do_flip[collected_N] == 1
      
      --local oi = valid_oi[{torch.random(valid_oi:nElement())}]
      local _, oi = torch.min(self.cat_counter:index(1, gtlabels:index(1, valid_oi):long()), 1)
      oi = valid_oi[oi[1]]
      local oi_start_idx = start_idx[oi]
      local oi_end_idx = end_idx[oi]
      
      local fi_start
      if not self.non_redundant_sampling then
        fi_start = torch.random(oi_start_idx, oi_end_idx - min_len + 1)
      else      
        local hit = self.sampling_hist[video_name][{{oi_start_idx, oi_end_idx - min_len + 1}}]
        local randperb = torch.randperm(hit:nElement()):type('torch.LongTensor')
        local randperb_hit = hit:index(1, randperb)
        local _, min_idx = torch.min(randperb_hit, 1)
        min_idx = randperb[min_idx[1]]
        hit[min_idx] = hit[min_idx] + 1
        fi_start = oi_start_idx + min_idx - 1
      end
      
      local fi_end = fi_start + min_len - 1
      for _, frame_idx in ipairs(torch.range(fi_start, fi_end, self.frame_stride):totable()) do
        local frame_valid_oi = torch.nonzero(indicator[frame_idx]):view(-1)
        
        -- get ground truth boxes
        local gtboxes = torch.FloatTensor(frame_valid_oi:nElement(), 5):zero()
        for ii, oii in ipairs(frame_valid_oi:totable()) do
          --local tmp_idx = torch.nonzero(obj[oii].boxes[{{}, 2}]:eq(frame_idx)):view(-1)
          local tmp_idx = frame_idx - obj[oii].start_frame + 1
          gtboxes[{ii, {1, 4}}]:copy(obj[oii].boxes[{tmp_idx, {3, 6}}])
          gtboxes[{ii, 5}] = oii
        end
        
        -- get proposals
        local rois, roi_scores = self:slice_prop(prop, frame_idx)
        
        if flip then
          rois = utils.flipBoxes_01(rois)
          gtboxes = utils.flipBoxes_01(gtboxes)
        end
        
        -- match rois and gtboxes
        table.insert(vid_box_coll, self:organize_boxes(gtboxes, rois, 
                      gtlabels:index(1, frame_valid_oi), #obj))
      end
      table.insert(box_coll, vid_box_coll)
      
      -- get image
      local frame_seq = torch.range(fi_start, fi_end, self.frame_stride)
      local images, im_sizes
      images, im_sizes = self:getImages(video_name, frame_seq, do_flip[collected_N], 1, self.spec_im_size)
      table.insert(image_coll, images)
      table.insert(im_size_coll, im_sizes)
    end
  end
  
  -- put all images into a giant tensor
  local images = self:squeeze_im_to_tensor(image_coll, im_size_coll)
  
  -- get rois and regression target
  local rois, labels, bboxregr_vals = self:gen_rois(box_coll, im_size_coll)
  
  ---- DIAGONOZE DISPLAY
  if DEBUG then
    local qtwidget = require 'qtwidget'
    local win = qtwidget.newwindow(images:size(4), images:size(3))  
    for debug_idx = 1, rois:size(1) do
      local debug_label = labels[debug_idx]
      if debug_label > 1 then
        local debug_img_idx = rois[debug_idx][1]
        local out = bboxregr_vals[debug_idx]:narrow(1,(debug_label-1)*4 + 1,4)
        out:cmul(self.bbox_regr.std):add(1,self.bbox_regr.mean)
        local raw_box = rois[{debug_idx, {2, 5}}]
        utils.convertFrom(out, raw_box, out)
        local x1 = out[1]
        local y1 = out[2]
        local x2 = out[3]
        local y2 = out[4]
        image.display({image = images[{debug_img_idx, {}, {}, {}}], win = win})
        win:fill()
        win:rectangle(x1, y1, x2-x1+1, y2-y1+1)
        win:stroke()
        print(string.format('cat:%s', self.cat_id_to_name[debug_label]))
        print('-----')
      end
    end
  end
  
  if self.seq_center then
    assert(self.batch_T % 2 == 1, 'You need to have odd number of frames to do seq_center mode.')
    local center_frm_idx = (self.batch_T - 1) / 2 + 1
    local center_idx = rois[{{}, 1}]:eq(center_frm_idx):nonzero():view(-1)
    rois = rois:index(1, center_idx)
    labels = labels:index(1, center_idx)
    bboxregr_vals = bboxregr_vals:index(1, center_idx)
  end
  
  
  if self.parallel_roi_batch > 1 then
    local roi_n = rois:size(1)
    local img_n = images:size(1)
    rois = torch.cat({rois, torch.range(1, roi_n):float():view(roi_n, 1)}, 2)
    local batch_n = img_n / self.parallel_roi_batch
    local roi_img_idx = rois[{{}, 1}]
    local roi_by_img = {}
    local roi_count_by_img = {}
    for idx = 1, img_n do
      local val_roi_idx = roi_img_idx:eq(idx):nonzero()
      assert(val_roi_idx:nElement() > 0, 'It cant be true that there is no roi at all for an image.')
      val_roi_idx = val_roi_idx:view(-1)
      roi_by_img[idx] = rois:index(1, val_roi_idx)
      roi_count_by_img[idx] = roi_by_img[idx]:size(1)
    end
    roi_count_by_img = torch.FloatTensor(roi_count_by_img)
    local max_rois = torch.max(roi_count_by_img)
    for idx = 1, img_n do
      local mod_idx = (idx - 1) % batch_n + 1
      roi_by_img[idx][{{}, 1}]:fill(mod_idx)
      if roi_count_by_img[idx] < max_rois then
        local fill_n = max_rois - roi_count_by_img[idx]
        local candTensor 
        if fill_n > roi_by_img[idx]:size(1) then
          local K = math.ceil(fill_n / roi_by_img[idx]:size(1))
          candTensor = torch.repeatTensor(roi_by_img[idx], K, 1) 
        else
          candTensor = roi_by_img[idx]
        end
        roi_by_img[idx] = torch.cat({candTensor[{{1, fill_n}, {}}], roi_by_img[idx]}, 1)
      end
    end
    rois = torch.cat(roi_by_img, 1)
    local sel_idx = rois[{{}, 6}]:long()
    rois = rois[{{}, {1,5}}]
    labels = labels:index(1, sel_idx)
    bboxregr_vals = bboxregr_vals:index(1, sel_idx)
  end
  
  local batches = {images, rois}
  local targets = {labels, {labels, bboxregr_vals}, g_donkey_idx}
  
  -- stats
  local counts = torch.histc(labels:float(), self.class_num+1, 1, self.class_num+1)
  self.cat_counter = self.cat_counter + counts
  
  return batches, targets
end


function BatchProviderVID:mine_hard_neg(scan_score, gt_class, opt)
  self.logsoftmax = self.logsoftmax or nn.LogSoftMax():cuda()
  local backprop_fg_num = math.floor(opt.backprop_batch_size * opt.fg_fraction)
  local backprop_bg_num = opt.backprop_batch_size - backprop_fg_num
  local prob = self.logsoftmax:forward(scan_score)
  local flat_idx = gt_class + torch.range(0, opt.num_classes * (prob:size(1) - 1), opt.num_classes):cuda()
  local gt_prob = prob:view(-1):index(1, flat_idx:long())
  local pos_idx = gt_class:gt(1):nonzero()
  if pos_idx:nElement() > 0 then
    pos_idx = pos_idx:view(-1)
    local pos_n = pos_idx:nElement()
    local sel_idx = torch.randperm(pos_n):narrow(1, 1, math.min(backprop_fg_num, pos_n))
    pos_idx = pos_idx:index(1, sel_idx:long())
  end
  local neg_idx = gt_class:eq(1):nonzero()
  if neg_idx:nElement() > 0 then
    neg_idx = neg_idx:view(-1)
    local y, i = torch.sort(gt_prob:index(1, neg_idx), 1)  
    i = i:narrow(1, 1, math.min(backprop_bg_num, neg_idx:nElement()))
    neg_idx = neg_idx:index(1, i)
  end
  local final_idx
  if pos_idx:nElement() > 0 and neg_idx:nElement() > 0 then 
    final_idx = torch.cat({pos_idx, neg_idx}, 1):long()
  elseif pos_idx:nElement() > 0 then
    final_idx = pos_idx:long()
  elseif neg_idx:nElement() > 0 then
    final_idx = neg_idx:long()
  else
    assert(false, 'Unexpected.')
  end  
  return final_idx
end

function BatchProviderVID:mine_hard_neg_v2(scan_box, scan_score, gt_class, opt)
  self.logsoftmax = self.logsoftmax or nn.LogSoftMax():cuda()
  local N = scan_score:size(1)
  local prob = self.logsoftmax:forward(scan_score)
  local flat_idx = gt_class + torch.range(0, opt.num_classes * (N - 1), opt.num_classes):cuda()
  local gt_prob = prob:view(-1):index(1, flat_idx:long())
  local sortval, sortidx = torch.sort(gt_prob, 1)
  local box = torch.FloatTensor(N, 5)
  box:narrow(2, 1, 4):copy(scan_box:index(1, sortidx))
  box:select(2, 5):copy(-sortval)
  local final_idx = utils.nms_dense(box, opt.ohem_nms_thresh)
  final_idx = sortidx:index(1, final_idx)
  final_idx = final_idx:narrow(1, 1, math.min(opt.backprop_batch_size, final_idx:nElement()))
  return final_idx
end

