--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local utils = paths.dofile'utils.lua'
local mu = paths.dofile'myutils.lua'
local ImageDetect = torch.class('fbcoco.ImageDetect')

function ImageDetect:__init(model, transformer, scale, max_size)
   assert(model, 'must provide model!')
   assert(transformer, 'must provide transformer!')
   self.model = model
   self.image_transformer = transformer
   self.scale = scale or {600}
   self.max_size = max_size or 1000
   self.sm = nn.SoftMax():cuda()
end

local function getImages(self,images,im)
   local num_scales = #self.scale

   local imgs = {}
   local im_sizes = {}
   local im_scales = {}

   im = self.image_transformer:forward(im)
   local channel = im:size(1)

   local im_size = im[1]:size()
   local im_size_min = math.min(im_size[1],im_size[2])
   local im_size_max = math.max(im_size[1],im_size[2])
   for i=1,num_scales do
      local im_scale = self.scale[i]/im_size_min
      if torch.round(im_scale*im_size_max) > self.max_size then
         im_scale = self.max_size/im_size_max
      end
      local im_s = {im_size[1]*im_scale,im_size[2]*im_scale}
      table.insert(imgs,image.scale(im,im_s[2],im_s[1]))
      table.insert(im_sizes,im_s)
      table.insert(im_scales,im_scale)
   end
   -- create single tensor with all images, padding with zero for different sizes
   im_sizes = torch.IntTensor(im_sizes)
   local max_shape = im_sizes:max(1)[1]
   images:resize(num_scales,channel,max_shape[1],max_shape[2]):zero()
   for i=1,num_scales do
      images[i][{{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}]:copy(imgs[i])
   end
   return im_scales
end

local function project_im_rois(im_rois,scales)
   local levels
   local rois = torch.FloatTensor()
   if #scales > 1 then
      assert(false, 'Multiple-scale mode is not properly implemented.')
      local scales = torch.FloatTensor(scales)
      local widths = im_rois[{{},3}] - im_rois[{{},1}] + 1
      local heights = im_rois[{{},4}] - im_rois[{{}, 2}] + 1

      local areas = widths * heights
      local scaled_areas = areas:view(-1,1) * torch.pow(scales:view(1,-1),2)
      local diff_areas = torch.abs(scaled_areas - 224 * 224)
      levels = select(2, diff_areas:min(2))
   else
      levels = torch.FloatTensor()
      rois:resize(im_rois:size(1),5)
      rois[{{},1}]:fill(1)
      rois[{{},{2,5}}]:copy(im_rois):add(-1):mul(scales[1]):add(1)
   end
   return rois
end

local function recursiveSplit(x, bs, dim)
   if type(x) == 'table' then
      local res = {}
      for k,v in pairs(x) do
         local tmp = v:split(bs,dim)
         for i=1,#tmp do
            if not res[i] then res[i] = {} end
            res[i][k] = tmp[i]
         end
      end
      return res
   else
      return x:split(bs, dim)
   end
end

function ImageDetect:memoryEfficientForward(model, input, bs, recompute_features)
   local images = input[1]
   local rois = input[2]
   local recompute_features = recompute_features == nil and true or recompute_features
   assert(model.output[1]:numel() > 0)

   local rest = nn.Sequential()
   for i=2,#model.modules do rest:add(model:get(i)) end
   local final = model:get(#model.modules)

   -- assuming the net has bbox regression part
   self.output = self.output or {torch.CudaTensor(), torch.CudaTensor()}
   local num_classes = self.model.output[1]:size(2)
   self.output[1]:resize(rois:size(1), num_classes)
   self.output[2]:resize(rois:size(1), num_classes * 4)

   if recompute_features then
      model:get(1):forward{images,rois}
   else
      model:get(1).output[2] = rois
   end

   local features = model:get(1).output
   assert(features[2]:size(1) == rois:size(1))

   local roi_split = features[2]:split(bs,1)
   local output1_split = self.output[1]:split(bs,1)
   local output2_split = self.output[2]:split(bs,1)

   for i,v in ipairs(roi_split) do
      local out = rest:forward({features[1], v})
      output1_split[i]:copy(out[1])
      output2_split[i]:copy(out[2])
   end

   local function test()
      local output_full = model:forward({images,rois})

      local output_split = self.output
      assert((output_full[1] - output_split[1]):abs():max() == 0)
      assert((output_full[2] - output_split[2]):abs():max() == 0)
   end
   --test()
   return self.output
end

function ImageDetect:computeRawOutputs(im, boxes, min_images, recompute_features)
   self.model:evaluate()

   local inputs = {torch.FloatTensor(),torch.FloatTensor()}
   local im_scales = getImages(self,inputs[1],im)
   inputs[2] = project_im_rois(boxes,im_scales)
   if min_images then
      assert(inputs[1]:size(1) == 1)
      inputs[1] = inputs[1]:expand(min_images, inputs[1]:size(2), inputs[1]:size(3), inputs[1]:size(4))
   end

   self.inputs_cuda = self.inputs_cuda or {torch.CudaTensor(),torch.CudaTensor()}
   self.inputs_cuda[1]:resize(inputs[1]:size()):copy(inputs[1])
   self.inputs_cuda[2]:resize(inputs[2]:size()):copy(inputs[2])

   return self.model:forward(self.inputs_cuda)
end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect(im, boxes, min_images, recompute_features)
   self.model:evaluate()

   local inputs = {torch.FloatTensor(),torch.FloatTensor()}
   local im_scales = getImages(self,inputs[1],im)
   inputs[2] = project_im_rois(boxes,im_scales)
   if min_images then
      assert(inputs[1]:size(1) == 1)
      inputs[1] = inputs[1]:expand(min_images, inputs[1]:size(2), inputs[1]:size(3), inputs[1]:size(4))
   end

   self.inputs_cuda = self.inputs_cuda or {torch.CudaTensor(),torch.CudaTensor()}
   self.inputs_cuda[1]:resize(inputs[1]:size()):copy(inputs[1])
   self.inputs_cuda[2]:resize(inputs[2]:size()):copy(inputs[2])

   local output0 = self:memoryEfficientForward(self.model, self.inputs_cuda, 500, recompute_features)
   --local output0 = self.model:forward(self.inputs_cuda)
  
   local class_values, bbox_values
   if torch.type(output0) == 'table' then
      class_values= output0[1]
      bbox_values = output0[2]:float()
      for i,v in ipairs(bbox_values:split(4,2)) do
         utils.convertFrom(v,boxes,v)
      end
   else
      class_values = output0
   end
   if not self.model.noSoftMax then
      class_values = self.sm:forward(class_values)
   end
   return class_values:float(), bbox_values
end

---------------------------------------------

function ImageDetect:detect_VID_LONGMEM(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  assert(det_T%2==1, 'det_T must be odd number.')
  
  -- Assume we can hold all conv map in memory
  local conv_stack = self.model.conv_stack
  local conv_maps = {}
  local B = math.ceil(T / det_T)
  local start_ptr = 1
  for bIdx = 1, B do
    local end_ptr = math.min(start_ptr + det_T - 1, T)
    local batch_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
    for tmpidx = 1, batch_im:size(1) do
      local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
      tmp_im:copy(self.image_transformer:forward(tmp_im))
    end
    batch_im = batch_im:cuda()
    local output = conv_stack:forward(batch_im):clone()
    table.insert(conv_maps, output)
    start_ptr = end_ptr + 1
  end
  conv_maps = torch.cat(conv_maps, 1)
  
  -- set STMM N and T
  local default_N, default_T = utils.set_NT(self.model.STMM_stack, 1, det_T)   

  -- forward, do left->right and right->left individually
  local B = math.ceil(T / det_T)
  local start_ptr, inv_start_ptr = 1, T
  local STMMs = self.model.STMM_stack:findModules('nn.STMM')
  assert(#STMMs == 2, 'You sure have only one layer?')
  local left2right = STMMs[1]
  local right2left = STMMs[2]
  local left2right_mem, right2left_mem, left2right_coll, right2left_coll = nil, nil, {}, {}
  for bidx = 1, B do
    collectgarbage()
    
    -- left -> right
    local end_ptr = math.min(start_ptr + det_T - 1, T)
    local len = end_ptr - start_ptr + 1
    local cur_conv_maps = conv_maps[{{start_ptr, end_ptr}, {}, {}, {}}]  
    utils.set_NT(left2right, 1, len)
    local left2right_output
    if left2right_mem then
      left2right_output = left2right:forward({cur_conv_maps, left2right_mem}):clone()
    else
      left2right_output = left2right:forward(cur_conv_maps):clone()
    end
    left2right_mem = left2right_output[{{len}, {}, {}, {}}]
    table.insert(left2right_coll, left2right_output)
    
    -- right -> left
    local inv_end_ptr = math.max(inv_start_ptr - det_T + 1, 1)
    local inv_len = inv_start_ptr - inv_end_ptr + 1
    local inv_seq = torch.range(inv_start_ptr, inv_end_ptr, -1):long()
    local cur_inv_conv_maps = conv_maps:index(1, inv_seq)
    utils.set_NT(right2left, 1, inv_len)
    local right2left_output
    if right2left_mem then
      right2left_output = right2left:forward({cur_inv_conv_maps, right2left_mem}):clone()
    else
      right2left_output = right2left:forward(cur_inv_conv_maps):clone()
    end
    right2left_mem = right2left_output[{{inv_len}, {}, {}, {}}]
    table.insert(right2left_coll, right2left_output)
    
    -- track counter
    start_ptr = end_ptr + 1
    inv_start_ptr = inv_end_ptr - 1 
  end
  left2right_coll = torch.cat(left2right_coll, 1)
  right2left_coll = torch.cat(right2left_coll, 1)
  
  -- revert back right2left_coll
  right2left_coll = right2left_coll:index(1, torch.range(T,1,-1):long()) 
  
  -- merge
  local stmm_maps = torch.cat({left2right_coll, right2left_coll}, 2)
  
  -- clear model state and collectgarbage
  conv_maps = nil
  collectgarbage()
   
  -- set back N and T
  utils.set_NT(self.model.STMM_stack, default_N, default_T)                          

  start_ptr = 1
  for bidx = 1, B do
    local end_ptr = math.min(start_ptr + det_T - 1, T)
    local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
    local cur_stmm_maps = stmm_maps[{{start_ptr, end_ptr}, {}, {}, {}}]      
    local score, bbox = self:conv2pred(cur_stmm_maps, cur_boxes, self.model.post_stack)
    collectgarbage()
    for tmpidx = 1, #cur_boxes do
      local glbidx = tmpidx + start_ptr - 1
      score_coll[glbidx] = score[tmpidx]
      bbox_coll[glbidx] = bbox[tmpidx]
    end
    start_ptr = end_ptr + 1
  end
  
  return score_coll, bbox_coll
end

---------------------------------------------

function ImageDetect:detect_VID_MULWINSIZE(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  
  if T <= det_T then
    score_coll, bbox_coll = self:detect_VID(im, boxes, min_images, recompute_features)
  else
    -- NOTE A set of window size
    local det_T_seq = {det_T, det_T-4, det_T-8}
    
    -- Assume we can hold all conv map in memory
    local conv_stack = self.model.conv_stack
    local conv_maps = {}
    local B = math.ceil(T / det_T)
    local start_ptr = 1
    for bIdx = 1, B do
      local end_ptr = math.min(start_ptr + det_T - 1, T)
      local batch_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
      for tmpidx = 1, batch_im:size(1) do
        local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
        tmp_im:copy(self.image_transformer:forward(tmp_im))
      end
      batch_im = batch_im:cuda()
      local output = conv_stack:forward(batch_im):clone()
      --output = output:float()
      table.insert(conv_maps, output)
      start_ptr = end_ptr + 1
    end
    conv_maps = torch.cat(conv_maps, 1)
  
    -- Fetch network
    self.top = self.top or nn.Sequential()
                            :add(nn.ParallelTable()
                              :add(self.model.STMM_stack)
                              :add(nn.Identity()))
                            :add(self.model.post_stack)
    local default_N, default_T = utils.set_NT(self.top, 1, 1) 
    
    -- Compute multiple window size
    for tIdx, t in ipairs(det_T_seq) do
      collectgarbage()
      -- Init container
      score_coll[tIdx] = {}
      bbox_coll[tIdx] = {}
      
      assert(t%2==1, 'det_T must be odd number.')
      local center_idx = (t - 1) / 2 + 1
      
      -- Set NT
      utils.set_NT(self.top, 1, t) 
      
      -- Compute STMM output
      for start_ptr = 1, T - t + 1 do        
        local end_ptr = math.min(start_ptr + t - 1, T)
        local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
        local cur_conv_maps = conv_maps[{{start_ptr, end_ptr}, {}, {}, {}}]      
        local score, bbox = self:conv2pred(cur_conv_maps, cur_boxes, self.top)
        collectgarbage()
        if start_ptr == 1 then
          for tmpidx = 1, center_idx do
            local glbidx = tmpidx + start_ptr - 1
            score_coll[tIdx][glbidx] = score[tmpidx]
            bbox_coll[tIdx][glbidx] = bbox[tmpidx]
          end
        elseif start_ptr == T-t+1 then
          for tmpidx = center_idx, t do
            local glbidx = tmpidx + start_ptr - 1
            score_coll[tIdx][glbidx] = score[tmpidx]
            bbox_coll[tIdx][glbidx] = bbox[tmpidx]
          end
        else
          local glbidx = center_idx + start_ptr - 1
          score_coll[tIdx][glbidx] = score[center_idx]
          bbox_coll[tIdx][glbidx] = bbox[center_idx]
        end
      end
    end
    
    -- Merge different win size
    local score_coll_flat, bbox_coll_flat = {}, {}
    for frame_idx = 1, T do
      local score_tmp, bbox_tmp = {}, {}
      for tIdx = 1, #det_T_seq do
        score_tmp[tIdx] = score_coll[tIdx][frame_idx]
        bbox_tmp[tIdx] = bbox_coll[tIdx][frame_idx]
      end
      score_coll_flat[frame_idx] = torch.cat(score_tmp, 1)
      bbox_coll_flat[frame_idx] = torch.cat(bbox_tmp, 1)
    end
    score_coll = score_coll_flat
    bbox_coll = bbox_coll_flat
    
    -- Set NT
    utils.set_NT(self.top, default_N, default_T) 
  end
  
  return score_coll, bbox_coll
end


function ImageDetect:detect_VID_LAST(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  assert(det_T%2==1, 'det_T must be odd number.')
  
  if T <= det_T then
    score_coll, bbox_coll = self:detect_VID(im, boxes, min_images, recompute_features)
  else
    -- Assume we can hold all conv map in memory
    local conv_stack = self.model.conv_stack
    local conv_maps = {}
    local B = math.ceil(T / det_T)
    local start_ptr = 1
    for bIdx = 1, B do
      local end_ptr = math.min(start_ptr + det_T - 1, T)
      local batch_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
      for tmpidx = 1, batch_im:size(1) do
        local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
        tmp_im:copy(self.image_transformer:forward(tmp_im))
      end
      batch_im = batch_im:cuda()
      local output = conv_stack:forward(batch_im):clone()
      --output = output:float()
      table.insert(conv_maps, output)
      start_ptr = end_ptr + 1
    end
    conv_maps = torch.cat(conv_maps, 1)
    
    -- Compute STMM output
    self.top = self.top or nn.Sequential()
                            :add(nn.ParallelTable()
                              :add(self.model.STMM_stack)
                              :add(nn.Identity()))
                            :add(self.model.post_stack)

    for start_ptr = 1, T - det_T + 1 do
      -- clear state of the model
      --self.top:clearState()
      
      local end_ptr = math.min(start_ptr + det_T - 1, T)
      local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
      local cur_conv_maps = conv_maps[{{start_ptr, end_ptr}, {}, {}, {}}]      
      local score, bbox = self:conv2pred(cur_conv_maps, cur_boxes, self.top)
      collectgarbage()
      if start_ptr == 1 then
        for tmpidx = 1, det_T do
          local glbidx = tmpidx + start_ptr - 1
          score_coll[glbidx] = score[tmpidx]
          bbox_coll[glbidx] = bbox[tmpidx]
        end
      else
        local glbidx = det_T + start_ptr - 1
        score_coll[glbidx] = score[det_T]
        bbox_coll[glbidx] = bbox[det_T]
      end
    end
  end
  
  return score_coll, bbox_coll
end

---------------------------------------------

function ImageDetect:detect_VID_NO_OVERLAP(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  
  self.model:clearState()
  
  local CONV_FORWARD_T = 4
  local conv_stack = self.model.conv_stack
  local conv_maps = {}
  local B = math.ceil(T / CONV_FORWARD_T)
  local start_ptr = 1
  for bIdx = 1, B do
    local end_ptr = math.min(start_ptr + CONV_FORWARD_T - 1, T)
    local batch_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
    for tmpidx = 1, batch_im:size(1) do
      local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
      tmp_im:copy(self.image_transformer:forward(tmp_im))
    end
    batch_im = batch_im:cuda()
    local output = conv_stack:forward(batch_im)
    output = output:float()
    table.insert(conv_maps, output)
    start_ptr = end_ptr + 1
  end
  conv_maps = torch.cat(conv_maps, 1)
  

  if T > det_T then
    local B = math.ceil(T / det_T)
    local start_ptr = 1
    for bIdx = 1, B do      
      local end_ptr = math.min(start_ptr + det_T - 1, T)
      local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
      local cur_conv_maps = conv_maps[{{start_ptr, end_ptr}, {}, {}, {}}]     
      cur_conv_maps = cur_conv_maps:cuda() 
      
      local only_center = false
      local score, bbox = self:conv2pred(cur_conv_maps, cur_boxes, 
                          self.model.STMM_stack, self.model.post_stack, only_center)
      collectgarbage()
      for tmpidx = 1, T do
        local glbidx = tmpidx + start_ptr - 1
        score_coll[glbidx] = score[tmpidx]
        bbox_coll[glbidx] = bbox[tmpidx]
      end
      start_ptr = end_ptr + 1
    end
  else
    conv_maps = conv_maps:cuda()
    local score, bbox = self:conv2pred(conv_maps, boxes, 
                        self.model.STMM_stack, self.model.post_stack, false)
    collectgarbage()
    for tmpidx = 1, T do
      score_coll[tmpidx] = score[tmpidx]
      bbox_coll[tmpidx] = bbox[tmpidx]
    end
  end
  
  return score_coll, bbox_coll
end

function ImageDetect:detect_VID_CENTER(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  assert(det_T%2==1, 'det_T must be odd number.')
  local center_idx = (det_T - 1) / 2 + 1
  
  self.model:clearState()
  
  if false then
  --if T <= det_T then
    score_coll, bbox_coll = self:detect_VID(
                              im, boxes, min_images, recompute_features)
  else
    local CONV_FORWARD_T = 1
    local conv_stack = self.model.conv_stack
    local conv_maps = {}
    local B = math.ceil(T / CONV_FORWARD_T)
    local start_ptr = 1
    for bIdx = 1, B do
      local end_ptr = math.min(start_ptr + CONV_FORWARD_T - 1, T)
      local batch_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
      for tmpidx = 1, batch_im:size(1) do
        local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
        tmp_im:copy(self.image_transformer:forward(tmp_im))
      end
      batch_im = batch_im:cuda()
      local output = conv_stack:forward(batch_im)
      output = output:float()
      table.insert(conv_maps, output)
      start_ptr = end_ptr + 1
    end
    conv_maps = torch.cat(conv_maps, 1)
    

    --print('Start compute detection.')
    if T > det_T then
      for start_ptr = 1, T - det_T + 1 do
        local end_ptr = math.min(start_ptr + det_T - 1, T)
        local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
        local cur_conv_maps = conv_maps[{{start_ptr, end_ptr}, {}, {}, {}}]     
        cur_conv_maps = cur_conv_maps:cuda() 
        
        local only_center = true
        if start_ptr == 1 or start_ptr == T - det_T + 1 then
          only_center = false
        end
        
        local score, bbox = self:conv2pred(cur_conv_maps, cur_boxes, 
                            self.model.STMM_stack, self.model.post_stack, only_center)
        collectgarbage()
        
        if start_ptr == 1 then
          for tmpidx = 1, center_idx do
            local glbidx = tmpidx + start_ptr - 1
            score_coll[glbidx] = score[tmpidx]
            bbox_coll[glbidx] = bbox[tmpidx]
          end
        elseif start_ptr == T-det_T+1 then
          for tmpidx = center_idx, det_T do
            local glbidx = tmpidx + start_ptr - 1
            score_coll[glbidx] = score[tmpidx]
            bbox_coll[glbidx] = bbox[tmpidx]
          end
        else
          local glbidx = center_idx + start_ptr - 1
          score_coll[glbidx] = score[center_idx]
          bbox_coll[glbidx] = bbox[center_idx]
        end
            
      end
    else
      conv_maps = conv_maps:cuda()
      local score, bbox = self:conv2pred(conv_maps, boxes, 
                          self.model.STMM_stack, self.model.post_stack, false)
      collectgarbage()
      for tmpidx = 1, T do
        score_coll[tmpidx] = score[tmpidx]
        bbox_coll[tmpidx] = bbox[tmpidx]
      end
    end
    --print('Done compute detection.')
  end
  
  return score_coll, bbox_coll
end

function ImageDetect:detect_VID_CENTER_OVERLAP(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  assert(det_T%2==1, 'det_T must be odd number.')
  local center_idx = (det_T - 1) / 2 + 1
  self.model:clearState()
  
  local CONV_FORWARD_T = 1
  local conv_stack = self.model.conv_stack
  local conv_maps = {}
  local B = math.ceil(T / CONV_FORWARD_T)
  local start_ptr = 1
  for bIdx = 1, B do
    local end_ptr = math.min(start_ptr + CONV_FORWARD_T - 1, T)
    local batch_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
    for tmpidx = 1, batch_im:size(1) do
      local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
      tmp_im:copy(self.image_transformer:forward(tmp_im))
    end
    batch_im = batch_im:cuda()
    local output = conv_stack:forward(batch_im)
    output = output:float()
    table.insert(conv_maps, output)
    start_ptr = end_ptr + 1
  end
  conv_maps = torch.cat(conv_maps, 1)
  
  if T > det_T then
    for start_ptr = 1, T - det_T + 1 do
      -- clear state of the model
      --self.top:clearState()
      local end_ptr = math.min(start_ptr + det_T - 1, T)
      local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
      local cur_conv_maps = conv_maps[{{start_ptr, end_ptr}, {}, {}, {}}]     
      cur_conv_maps = cur_conv_maps:cuda() 
      
      local only_center = false
      
      -- conv2pred_iter_replace or conv2pred_iter
      local score, bbox = self:conv2pred_iter(cur_conv_maps, cur_boxes, 
                          self.model.STMM_stack, self.model.post_stack, only_center)
      collectgarbage()
      for tmpidx = 1, #score do
        local glbidx = tmpidx + start_ptr - 1
        if score_coll[glbidx] and score_coll[glbidx]:nElement() > 0 then
          score_coll[glbidx] = torch.cat({score[tmpidx], score_coll[glbidx]}, 1)
        else
          score_coll[glbidx] = score[tmpidx]
        end
        if bbox_coll[glbidx] and bbox_coll[glbidx]:nElement() > 0 then
          bbox_coll[glbidx] = torch.cat({bbox[tmpidx], bbox_coll[glbidx]}, 1)
        else
          bbox_coll[glbidx] = bbox[tmpidx]
        end
      end
    end
  else
    local cur_conv_maps = conv_maps:cuda()
    -- conv2pred_iter_replace or conv2pred_iter
    local score, bbox = self:conv2pred_iter(cur_conv_maps, boxes, 
                        self.model.STMM_stack, self.model.post_stack, false)
    collectgarbage()
    for tmpidx = 1, T do
      score_coll[tmpidx] = score[tmpidx]
      bbox_coll[tmpidx] = bbox[tmpidx]
    end
  end
  --print('Done compute detection.')
  
  return score_coll, bbox_coll, conv_maps
end

function ImageDetect:detect_VID_CENTER_OVERLAP_MULTISCALE(im, boxes, det_T, min_images, recompute_features)
  self.model:evaluate()
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  assert(T == #boxes, '#im does not equate to #boxes.')
  assert(det_T%2==1, 'det_T must be odd number.')
  local center_idx = (det_T - 1) / 2 + 1
  
  self.model:clearState()
  if false then
  --if T <= det_T then
    score_coll, bbox_coll = self:detect_VID(
                              im, boxes, min_images, recompute_features)
  else
    local scales = {270, 360, 540}
    local ratios = {}
    local hgt, wid = im:size(3), im:size(4)
    local conv_maps = {}
    local CONV_FORWARD_T = 3
    local conv_stack = self.model.conv_stack
    local B = math.ceil(T / CONV_FORWARD_T)
    for scale_idx, scale in ipairs(scales) do
      local cur_conv_maps = {}
      local im_size_min = math.min(hgt, wid)
      local im_size_max = math.max(hgt, wid)
      ratios[scale_idx] = scale / im_size_min
      local cur_hgt, cur_wid = hgt*ratios[scale_idx], wid*ratios[scale_idx] 
      local cur_im = torch.FloatTensor(im:size(1), 3, cur_hgt, cur_wid)
      for idx = 1, im:size(1) do
        cur_im[idx]:copy(image.scale(im[idx], cur_wid, cur_hgt))
      end
      local start_ptr = 1
      for bIdx = 1, B do
        local end_ptr = math.min(start_ptr + CONV_FORWARD_T - 1, T)
        local batch_im = cur_im[{{start_ptr, end_ptr}, {}, {}, {}}]
        for tmpidx = 1, batch_im:size(1) do
          local tmp_im = batch_im[{tmpidx, {}, {}, {}}]
          tmp_im:copy(self.image_transformer:forward(tmp_im))
        end
        batch_im = batch_im:cuda()
        local output = conv_stack:forward(batch_im)
        output = output:float()
        table.insert(cur_conv_maps, output)
        start_ptr = end_ptr + 1
      end
      cur_conv_maps = torch.cat(cur_conv_maps, 1)
      conv_maps[scale_idx] = cur_conv_maps
    end
    
    
    if T > det_T then
      for start_ptr = 1, T - det_T + 1 do
        -- clear state of the model
        --self.top:clearState()
        local end_ptr = math.min(start_ptr + det_T - 1, T)
        local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
        local cur_conv_maps = {}
        for idx = 1, #conv_maps do
          cur_conv_maps[idx] = conv_maps[idx][{{start_ptr, end_ptr}, {}, {}, {}}]:cuda()
        end
        local only_center = false
        local score, bbox = self:conv2pred_mulscale(cur_conv_maps, cur_boxes, ratios, 
                            self.model.STMM_stack, self.model.post_stack, only_center)
        collectgarbage()
        
        for tmpidx = 1, #score do
          local glbidx = tmpidx + start_ptr - 1
          if score_coll[glbidx] and score_coll[glbidx]:nElement() > 0 then
            score_coll[glbidx] = torch.cat({score[tmpidx], score_coll[glbidx]}, 1)
          else
            score_coll[glbidx] = score[tmpidx]
          end
          if bbox_coll[glbidx] and bbox_coll[glbidx]:nElement() > 0 then
            bbox_coll[glbidx] = torch.cat({bbox[tmpidx], bbox_coll[glbidx]}, 1)
          else
            bbox_coll[glbidx] = bbox[tmpidx]
          end
        end     
      end
    else
      for idx = 1, #conv_maps do
        conv_maps[idx] = conv_maps[idx]:cuda()
      end
      local score, bbox = self:conv2pred_mulscale(conv_maps, boxes, ratios, 
                          self.model.STMM_stack, self.model.post_stack, false)
      collectgarbage()
      for tmpidx = 1, T do
        score_coll[tmpidx] = score[tmpidx]
        bbox_coll[tmpidx] = bbox[tmpidx]
      end
    end
    --print('Done compute detection.')
  end
  
  return score_coll, bbox_coll
end


function ImageDetect:detect_VID_SEQ(im, boxes, det_T, min_images, recompute_features)
  det_T = 1
  local score_coll = {}
  local bbox_coll = {}
  local T = im:size(1)
  -- assert(T % det_T == 0, 'Number of images must be multiple of detector capacity.')
  local B = math.ceil(T / det_T)
  local start_ptr = 1
  for bidx = 1, B do
    local end_ptr = math.min(start_ptr + det_T - 1, T)
    local cur_im = im[{{start_ptr, end_ptr}, {}, {}, {}}]
    local cur_boxes = {table.unpack(boxes, start_ptr, end_ptr)}
    local score, bbox = self:detect_VID(cur_im, cur_boxes, min_images, recompute_features)
    for tmpidx = 1, #score do
      local glbidx = tmpidx + start_ptr - 1
      score_coll[glbidx] = score[tmpidx]
      bbox_coll[glbidx] = bbox[tmpidx]
    end
    start_ptr = end_ptr + 1
  end
  return score_coll, bbox_coll
end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect_VID(im, boxes, min_images, recompute_features)
   assert(#self.model:findModules('nn.BBoxNorm') > 0, 'WARNING: No nn.BBoxNorm is not found in the model.')
   assert(#self.scale==1, 'Current implementation only supports testing with single scale.')
   self.model:evaluate()
   local H, W = im:size(3), im:size(4)
   local im_coll, box_coll = {}, {}
   local box_count, im_box_start = 0, {}
   for img_idx = 1, im:size(1) do
     local cur_inputs = {torch.FloatTensor(),torch.FloatTensor()}
     local cur_im = im[{img_idx, {}, {}, {}}]
     local cur_boxes = boxes[img_idx]
     local im_scales = getImages(self,cur_inputs[1],cur_im)
     cur_inputs[2] = project_im_rois(cur_boxes,im_scales)
     cur_inputs[2][{{}, 1}]:fill(img_idx)
     table.insert(im_coll, cur_inputs[1])
     table.insert(box_coll, cur_inputs[2])
     im_box_start[img_idx] = box_count + 1
     local cur_box_count = cur_inputs[2]:size(1)
     box_count = box_count + cur_box_count
   end
   table.insert(im_box_start, box_count + 1)
   im_coll = torch.cat(im_coll, 1)
   box_coll = torch.cat(box_coll, 1)
   local inputs = {}
   inputs[1] = im_coll
   inputs[2] = box_coll
   boxes = torch.cat(boxes, 1)
   
   -- some global vars
   local expanded_T = inputs[1]:size(1)
   
   -- expand (pad)
   if min_images then
    if inputs[1]:size(1) % min_images ~= 0 then
      local pad_T = min_images - inputs[1]:size(1) % min_images
      inputs[1] = torch.cat({inputs[1], inputs[1].new(pad_T, inputs[1]:size(2), 
                        inputs[1]:size(3), inputs[1]:size(4)):zero()}, 1)
      expanded_T = inputs[1]:size(1)
    end
   end
   
   self.inputs_cuda = self.inputs_cuda or {torch.CudaTensor(),torch.CudaTensor()}
   self.inputs_cuda[1]:resize(inputs[1]:size()):copy(inputs[1])
   self.inputs_cuda[2]:resize(inputs[2]:size()):copy(inputs[2])
   
   -- set STMM N and T
   local default_N, default_T = utils.set_NT(self.model, 1, expanded_T) 
   
   -- forward
   local output0 = self.model:forward(self.inputs_cuda)
   
   -- set back N and T
   utils.set_NT(self.model, default_N, default_T)
   
   local class_values, bbox_values
   if torch.type(output0) == 'table' then
      class_values= output0[1]
      bbox_values = output0[2]:float()
      for i,v in ipairs(bbox_values:split(4,2)) do
         utils.convertFrom(v,boxes,v)
      end
   else
      class_values = output0
   end
   if not self.model.noSoftMax then
      class_values = self.sm:forward(class_values)
   end
   class_values = class_values:float()
   
   -- pack for different image
   local class_values_coll, bbox_values_coll = {}, {}
   for img_idx = 1, im:size(1) do
    local start_idx = im_box_start[img_idx]
    local end_idx = im_box_start[img_idx+1]-1 or class_values:size(1)
    class_values_coll[img_idx] = class_values[{{start_idx, end_idx}, {}}]
    bbox_values_coll[img_idx] = bbox_values[{{start_idx, end_idx}, {}}]
   end
   
   return class_values_coll, bbox_values_coll
end


function ImageDetect:conv2pred_mulscale(conv_maps, boxes, ratios, STMM_model, post_model, only_center)
   assert(#post_model:findModules('nn.BBoxNorm') > 0, 'WARNING: No nn.BBoxNorm is not found in the post stack.')
   local ITER_NUM = 2
   local ITER_SCORE_THRESH = 0.05

   local T = conv_maps[1]:size(1)
   local center_idx = (T + 1) / 2
   assert(T == #boxes, 'Number of image should be equal to number of boxes.')
   local STMM_maps = {}
   -- set STMM N and T
   local default_N, default_T = utils.set_NT(STMM_model, 1, T)  
   for idx = 1, #conv_maps do
     STMM_maps[idx] = STMM_model:forward(conv_maps[idx]):clone()
   end
   -- set back N and T
   utils.set_NT(self.model, default_N, default_T)
   
   -- get the roi pooling module
   local ROI_GRID = 14
   self.roi_pool = self.roi_pool or inn.ROIPooling(ROI_GRID,ROI_GRID,1/16):cuda()
   
   local class_values_coll, bbox_values_coll = {}, {}
   for img_idx = 1, T do
     if not only_center or img_idx == center_idx then
       local cur_boxes = torch.FloatTensor(boxes[img_idx]:size(1), 5)
       cur_boxes:narrow(2, 2, 4):copy(boxes[img_idx])
       cur_boxes:select(2, 1):fill(1)
       local score_coll, box_coll = {}, {}
       for iter = 1, ITER_NUM do
         if cur_boxes and cur_boxes:nElement() > 0 then
           -- max-out
           local roi_feat
           local roi_num, feat_dim = cur_boxes:size(1), STMM_maps[1]:size(2)
           for idx = 1, #STMM_maps do
            local cur_scale_box = cur_boxes:clone()
            cur_scale_box:narrow(2, 2, 4):mul(ratios[idx])
            local cur_STMM_maps = STMM_maps[idx][{{img_idx},{},{},{}}]
            local tmp = self.roi_pool:forward({cur_STMM_maps, cur_scale_box:cuda()})
            tmp = tmp:view(1, roi_num, feat_dim, ROI_GRID, ROI_GRID)
            if idx == 1 then
              roi_feat = tmp:clone()
            else
              roi_feat:copy(torch.max(torch.cat({tmp, roi_feat}, 1), 1))
            end
            collectgarbage()
           end
           roi_feat = roi_feat:view(roi_num, feat_dim, ROI_GRID, ROI_GRID)
           
           local output = post_model:forward(roi_feat)
           local score, box_coef = output[1]:float(), output[2]:float()
           for i,v in ipairs(box_coef:split(4,2)) do
              utils.convertFrom(v,cur_boxes:narrow(2, 2, 4),v)
           end
           if not self.model.noSoftMax then
              if self.sm:type() ~= 'torch.FloatTensor' then
                self.sm = self.sm:float()
              end
              score:copy(self.sm:forward(score))
           end
           
           local box_coef_flat = box_coef:view(box_coef:size(1), -1, 4)
           box_coef_flat = box_coef_flat:narrow(2, 2, box_coef_flat:size(2) - 1):contiguous():view(-1, 4)
           local score_flat = score:narrow(2, 2, score:size(2) - 1):contiguous():view(-1)
           
           local idx = score_flat:view(-1):ge(ITER_SCORE_THRESH):nonzero()
           if idx:nElement() > 0 then
            idx = idx:view(-1)
            cur_boxes:resize(idx:nElement(), 5)
            cur_boxes:narrow(2, 2, 4):copy(box_coef_flat:index(1, idx))
            cur_boxes:select(2, 1):fill(1)
           else
            cur_boxes = torch.FloatTensor()
           end
           table.insert(score_coll, score)
           table.insert(box_coll, box_coef)
         end
       end
       if #score_coll > 0 and score_coll[1]:nElement() > 0 then
         score_coll = torch.cat(score_coll, 1)
         box_coll = torch.cat(box_coll, 1)
       end
       class_values_coll[img_idx] = score_coll
       bbox_values_coll[img_idx] = box_coll
     end
   end
   
   return class_values_coll, bbox_values_coll
end

function ImageDetect:conv2pred_iter(conv_maps, boxes, STMM_model, post_model, only_center)
   assert(#post_model:findModules('nn.BBoxNorm') > 0, 'WARNING: No nn.BBoxNorm is not found in the post stack.')
   
   local ITER_NUM = 2
   local ITER_SCORE_THRESH = 0.05

   local T = conv_maps:size(1)
   local center_idx = (T + 1) / 2
   assert(T == #boxes, 'Number of image should be equal to number of boxes.')
   -- set STMM N and T
   local default_N, default_T = utils.set_NT(STMM_model, 1, T)  
   -- forward
   local STMM_maps = STMM_model:forward(conv_maps)
   -- set back N and T
   utils.set_NT(self.model, default_N, default_T)

   local class_values_coll, bbox_values_coll = {}, {}
   for img_idx = 1, T do
     local cur_rois_N = boxes[img_idx]:size(1)
     local cur_rois = torch.FloatTensor(cur_rois_N, 5)
     cur_rois:select(2, 1):fill(1)
     cur_rois:narrow(2, 2, 4):copy(boxes[img_idx])
     if not only_center or img_idx == center_idx then
       local score_coll, box_coll = {}, {}
       for iter = 1, ITER_NUM do
        if cur_rois:nElement() > 0 then
         local output = post_model:forward({STMM_maps[{{img_idx}, {}, {}, {}}], cur_rois:cuda()})
         local score, box_coef = output[1]:float(), output[2]:float()
         for i,v in ipairs(box_coef:split(4,2)) do
            utils.convertFrom(v,cur_rois:narrow(2, 2, 4),v)
         end
         if not self.model.noSoftMax then
            if self.sm:type() ~= 'torch.FloatTensor' then
              self.sm = self.sm:float()
            end
            score:copy(self.sm:forward(score))
         end
         
         local box_coef_flat = box_coef:view(box_coef:size(1), -1, 4)
         box_coef_flat = box_coef_flat:narrow(2, 2, box_coef_flat:size(2) - 1):contiguous():view(-1, 4)
         local score_flat = score:narrow(2, 2, score:size(2) - 1):contiguous():view(-1)
         local idx = score_flat:view(-1):ge(ITER_SCORE_THRESH):nonzero()
         if idx:nElement() > 0 then
          idx = idx:view(-1)
          cur_rois:resize(idx:nElement(), 5)
          cur_rois:narrow(2, 2, 4):copy(box_coef_flat:index(1, idx))
          cur_rois:select(2, 1):fill(1)          
         else
          cur_rois = torch.FloatTensor() 
         end
         
         if score:nElement() > 0 then
          table.insert(score_coll, score)
         end
         if box_coef:nElement() > 0 then
          table.insert(box_coll, box_coef)
         end
        end
       end
       if #score_coll > 0 and score_coll[1]:nElement() > 0 then
         score_coll = torch.cat(score_coll, 1)
         box_coll = torch.cat(box_coll, 1)
       end
       class_values_coll[img_idx] = score_coll
       bbox_values_coll[img_idx] = box_coll
     end
   end
   
   return class_values_coll, bbox_values_coll
end



function ImageDetect:conv2pred_iter_replace(conv_maps, boxes, STMM_model, post_model, only_center)
   assert(#post_model:findModules('nn.BBoxNorm') > 0, 'WARNING: No nn.BBoxNorm is not found in the post stack.')   
   local ITER_NUM = 3
   local ITER_SCORE_THRESH = 0.025

   local T = conv_maps:size(1)
   local center_idx = (T + 1) / 2
   assert(T == #boxes, 'Number of image should be equal to number of boxes.')
   -- set STMM N and T
   local default_N, default_T = utils.set_NT(STMM_model, 1, T)  
   -- forward
   local STMM_maps = STMM_model:forward(conv_maps)
   -- set back N and T
   utils.set_NT(self.model, default_N, default_T)

   local class_values_coll, bbox_values_coll = {}, {}
   for img_idx = 1, T do
     local cur_rois_N = boxes[img_idx]:size(1)
     local cur_rois = torch.FloatTensor(cur_rois_N, 5)
     cur_rois:select(2, 1):fill(1)
     cur_rois:narrow(2, 2, 4):copy(boxes[img_idx])
     if not only_center or img_idx == center_idx then
       local score_coll, box_coll = {}, {}
       for iter = 1, ITER_NUM do
        if cur_rois:nElement() > 0 then
         local output = post_model:forward({STMM_maps[{{img_idx}, {}, {}, {}}], cur_rois:cuda()})
         local score, box_coef = output[1]:float(), output[2]:float()
         for i,v in ipairs(box_coef:split(4,2)) do
            utils.convertFrom(v,cur_rois:narrow(2, 2, 4),v)
         end
         if not self.model.noSoftMax then
            if self.sm:type() ~= 'torch.FloatTensor' then
              self.sm = self.sm:float()
            end
            score:copy(self.sm:forward(score))
         end
         local box_coef_unit = box_coef:narrow(2,1,4)  -- here we are assuming uniform box regression across categories
         local max_pos_score = torch.max(score:narrow(2,2,score:size(2)-1), 2)
         local idx = max_pos_score:view(-1):ge(ITER_SCORE_THRESH):nonzero()
         if idx:nElement() > 0 then
          idx = idx:view(-1)
          cur_rois:resize(idx:nElement(), 5)
          cur_rois:narrow(2, 2, 4):copy(box_coef_unit:index(1, idx))
          cur_rois:select(2, 1):fill(1)
          if iter ~= ITER_NUM then
            local keep_idx = torch.FloatTensor(score:size(1)):fill(1)
            keep_idx:indexFill(1, idx:long(), 0)
            keep_idx = keep_idx:nonzero()
            if keep_idx:nElement() > 0 then
              keep_idx = keep_idx:view(-1)
              score = score:index(1, keep_idx)
              box_coef = box_coef:index(1, keep_idx)
            else
              score = torch.FloatTensor()
              box_coef = torch.FloatTensor()
            end
          end
         else
          cur_rois = torch.FloatTensor() 
         end
         
         if score:nElement() > 0 then
          table.insert(score_coll, score)
         end
         if box_coef:nElement() > 0 then
          table.insert(box_coll, box_coef)
         end
        end
       end
       if #score_coll > 0 and score_coll[1]:nElement() > 0 then
         score_coll = torch.cat(score_coll, 1)
         box_coll = torch.cat(box_coll, 1)
       end
       class_values_coll[img_idx] = score_coll
       bbox_values_coll[img_idx] = box_coll
     end
   end
   
   return class_values_coll, bbox_values_coll
end



function ImageDetect:conv2pred(conv_maps, boxes, STMM_model, post_model, only_center)
   assert(#post_model:findModules('nn.BBoxNorm') > 0, 'WARNING: No nn.BBoxNorm is not found in the post stack.')
   local T = conv_maps:size(1)
   local center_idx = (T + 1) / 2
   assert(T == #boxes, 'Number of image should be equal to number of boxes.')
   -- set STMM N and T
   local default_N, default_T = utils.set_NT(STMM_model, 1, T)  
   -- forward
   local STMM_maps = STMM_model:forward(conv_maps)
   -- set back N and T
   utils.set_NT(self.model, default_N, default_T)

   local score_list, regcoef_list = {}, {}, {}
   local count, im_box_start = 0, {}
   for img_idx = 1, T do
     im_box_start[img_idx] = count + 1
     local cur_rois = torch.CudaTensor(boxes[img_idx]:size(1), 5)
     cur_rois:select(2, 1):fill(1)
     cur_rois:narrow(2, 2, 4):copy(boxes[img_idx])
     local roi_num = cur_rois:size(1)
     if not only_center then
       local output = post_model:forward({STMM_maps[{{img_idx}, {}, {}, {}}], cur_rois})
       score_list[img_idx] = output[1]:float()
       regcoef_list[img_idx] = output[2]:float()
     else
       if img_idx == center_idx then
         local output = post_model:forward({STMM_maps[{{img_idx}, {}, {}, {}}], cur_rois})
         score_list = output[1]:float()
         regcoef_list = output[2]:float()
       end
     end
     count = count + cur_rois:size(1)
   end
   
   local rois, score, regcoef
   if not only_center then
     rois = torch.cat(boxes, 1) 
     score = torch.cat(score_list, 1)
     regcoef = torch.cat(regcoef_list, 1)
   else
     rois = boxes[center_idx]
     score = score_list
     regcoef = regcoef_list
   end
   
   for i,v in ipairs(regcoef:split(4,2)) do
      utils.convertFrom(v,rois,v)
   end
   if not self.model.noSoftMax then
      if self.sm:type() ~= 'torch.FloatTensor' then
        self.sm = self.sm:float()
      end
      score = self.sm:forward(score)
   end
   
   -- pack for different image
   local class_values_coll, bbox_values_coll = {}, {}
   if not only_center then
     for img_idx = 1, T do
      local start_idx = im_box_start[img_idx]
      local end_idx
      if im_box_start[img_idx + 1] then
        end_idx = im_box_start[img_idx + 1] - 1 
      else
        end_idx = score:size(1)
      end
      class_values_coll[img_idx] = score[{{start_idx, end_idx}, {}}]:clone()
      bbox_values_coll[img_idx] = regcoef[{{start_idx, end_idx}, {}}]:clone()
     end
   else
     class_values_coll[center_idx] = score:clone()
     bbox_values_coll[center_idx] = regcoef:clone()
   end
   
   return class_values_coll, bbox_values_coll
end
