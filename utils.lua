--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

stringx = require('pl.stringx') -- must be global or threads will barf :(

local tnt = require 'torchnet'
local mu = require 'myutils'

local utils = {}

local ffi = require 'ffi'
ffi.cdef[[
void bbox_vote(THFloatTensor *res, THFloatTensor *nms_boxes, THFloatTensor *scored_boxes, float threshold);
void NMS(THFloatTensor *keep, THFloatTensor *scored_boxes, float overlap);
void compute_iou(THFloatTensor *result, THFloatTensor *a, THFloatTensor *b);
]]

-- resolve some path ambiguity
local fname = 'libnms.so'
local s = paths.thisfile(nil)
if s and s ~= "" then
  fname = paths.concat(paths.dirname(s),fname)
end

local ok, C = pcall(ffi.load, fname)
if not ok then
   os.execute'make'
   ok, C = pcall(ffi.load, fname)
   assert(ok, 'run make and check what is wrong')
end

function utils.compute_iou(a, b)
   a = a:contiguous()
   b = b:contiguous()
   local N_a = a:size(1)
   local N_b = b:size(1)
   local c = torch.FloatTensor(N_a, N_b):zero()
   C.compute_iou(c:cdata(), a:cdata(), b:cdata())
   return c
end

function utils.nms(boxes, overlap)
   local keep = torch.FloatTensor()
   C.NMS(keep:cdata(), boxes:cdata(), overlap)
   return keep
end

function utils.bbox_vote(nms_boxes, scored_boxes, overlap)
   local res = torch.FloatTensor()
   C.bbox_vote(res:cdata(), nms_boxes:cdata(), scored_boxes:cdata(), overlap)
   return res
end


--------------------------------------------------------------------------------
-- utility functions for the evaluation part
--------------------------------------------------------------------------------

function utils.joinTable(input,dim)
   local size = torch.LongStorage()
   local is_ok = false
   for i=1,#input do
      local currentOutput = input[i]
      if currentOutput:numel() > 0 then
         if not is_ok then
            size:resize(currentOutput:dim()):copy(currentOutput:size())
            is_ok = true
         else
            size[dim] = size[dim] + currentOutput:size(dim)
         end
      end
   end
   local output = input[1].new():resize(size)
   local offset = 1
   for i=1,#input do
      local currentOutput = input[i]
      if currentOutput:numel() > 0 then
         output:narrow(dim, offset,
         currentOutput:size(dim)):copy(currentOutput)
         offset = offset + currentOutput:size(dim)
      end
   end
   return output
end

--------------------------------------------------------------------------------

function utils.keep_top_k(boxes,top_k)
   local X = utils.joinTable(boxes,1)
   if X:numel() == 0 then
      return boxes, 0
   end
   local scores = X[{{},-1}]:sort(1,true)
   local thresh = scores[math.min(scores:numel(),top_k)]
   for i=1,#boxes do
      local bbox = boxes[i]
      if bbox:numel() > 0 then
         local idx = torch.range(1,bbox:size(1)):long()
         local keep = bbox[{{},-1}]:ge(thresh)
         idx = idx[keep]
         if idx:numel() > 0 then
            boxes[i] = bbox:index(1,idx)
         else
            boxes[i]:resize()
         end
      end
   end
   return boxes, thresh
end

--------------------------------------------------------------------------------
-- evaluation
--------------------------------------------------------------------------------



function utils.calibrate_box(box, hgt, wid)
  if box:nElement() > 4 then
    local x1 = box:select(2,1)
    local y1 = box:select(2,2)
    local x2 = box:select(2,3)
    local y2 = box:select(2,4)
    x1[x1:lt(1)] = 1
    y1[y1:lt(1)] = 1
    x2[x2:gt(wid)] = wid
    y2[y2:gt(hgt)] = hgt
  else
    local shape = box:size()
    box = box:view(-1)
    box[1] = math.max(box[1], 1)
    box[2] = math.max(box[2], 1)
    box[3] = math.min(box[3], wid)
    box[4] = math.min(box[4], hgt)
    box = box:view(shape)
  end
  return box
end

function utils.calibrate_box_01(box)
  if box:nElement() > 4 then
    local x1 = box:select(2,1)
    local y1 = box:select(2,2)
    local x2 = box:select(2,3)
    local y2 = box:select(2,4)
    x1[x1:lt(0)] = 0
    y1[y1:lt(0)] = 0
    x2[x2:gt(1)] = 1
    y2[y2:gt(1)] = 1
  else
    local shape = box:size()
    box = box:view(-1)
    box[1] = math.max(box[1], 0)
    box[2] = math.max(box[2], 0)
    box[3] = math.min(box[3], 1)
    box[4] = math.min(box[4], 1)
    box = box:view(shape)
  end
  return box
end

-- A version of overlap function that operates on the range of [0, 1]
function utils.boxoverlap_01(a,b)
   local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
   local x1 = a:select(2,1):clone()
   x1[x1:lt(b[1])] = b[1]
   local y1 = a:select(2,2):clone()
   y1[y1:lt(b[2])] = b[2]
   local x2 = a:select(2,3):clone()
   x2[x2:gt(b[3])] = b[3]
   local y2 = a:select(2,4):clone()
   y2[y2:gt(b[4])] = b[4]
   local w = x2-x1;
   local h = y2-y1;
   local inter = torch.cmul(w,h):float()
   local aarea = torch.cmul((a:select(2,3)-a:select(2,1)) ,
   (a:select(2,4)-a:select(2,2))):float()
   local barea = (b[3]-b[1]) * (b[4]-b[2]);
   -- intersection over union overlap
   local o = torch.cdiv(inter , (aarea+barea-inter))
   -- set invalid entries to 0 overlap
   o[w:lt(0)] = 0
   o[h:lt(0)] = 0
   return o
end

function utils.boxoverlap_01_set(a_set, b_set)
   local iou = torch.FloatTensor(a_set:size(1), b_set:size(1)):zero()
   for b_idx = 1, b_set:size(1) do
     local b = b_set[b_idx]
     local x1 = a_set:select(2,1):clone()
     x1[x1:lt(b[1])] = b[1]
     local y1 = a_set:select(2,2):clone()
     y1[y1:lt(b[2])] = b[2]
     local x2 = a_set:select(2,3):clone()
     x2[x2:gt(b[3])] = b[3]
     local y2 = a_set:select(2,4):clone()
     y2[y2:gt(b[4])] = b[4]
  
     local w = x2-x1;
     local h = y2-y1;
     local inter = torch.cmul(w,h):float()
     local aarea = torch.cmul((a_set:select(2,3)-a_set:select(2,1)), (a_set:select(2,4)-a_set:select(2,2))):float()
     local barea = (b[3]-b[1]) * (b[4]-b[2])
  
     -- intersection over union overlap
     local o = torch.cdiv(inter, (aarea+barea-inter))
     -- set invalid entries to 0 overlap
     o[w:lt(0)] = 0
     o[h:lt(0)] = 0
     iou[{{}, b_idx}]:copy(o)
   end
   return iou
end

function utils.boxoverlap_set(a_set, b_set)
   local iou = torch.FloatTensor(a_set:size(1), b_set:size(1)):zero()
   for b_idx = 1, b_set:size(1) do
     local b = b_set[b_idx]
     local x1 = a_set:select(2,1):clone()
     x1[x1:lt(b[1])] = b[1]
     local y1 = a_set:select(2,2):clone()
     y1[y1:lt(b[2])] = b[2]
     local x2 = a_set:select(2,3):clone()
     x2[x2:gt(b[3])] = b[3]
     local y2 = a_set:select(2,4):clone()
     y2[y2:gt(b[4])] = b[4]
  
     local w = x2-x1+1;
     local h = y2-y1+1;
     local inter = torch.cmul(w,h):float()
     local aarea = torch.cmul((a_set:select(2,3)-a_set:select(2,1)+1) ,
     (a_set:select(2,4)-a_set:select(2,2)+1)):float()
     local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);
  
     -- intersection over union overlap
     local o = torch.cdiv(inter , (aarea+barea-inter))
     -- set invalid entries to 0 overlap
     o[w:lt(0)] = 0
     o[h:lt(0)] = 0
     iou[{{}, b_idx}]:copy(o)
   end
   return iou
end


function utils.boxoverlap(a,b)
   local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
   local x1 = a:select(2,1):clone()
   x1[x1:lt(b[1])] = b[1]
   local y1 = a:select(2,2):clone()
   y1[y1:lt(b[2])] = b[2]
   local x2 = a:select(2,3):clone()
   x2[x2:gt(b[3])] = b[3]
   local y2 = a:select(2,4):clone()
   y2[y2:gt(b[4])] = b[4]

   local w = x2-x1+1;
   local h = y2-y1+1;
   local inter = torch.cmul(w,h):float()
   local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
   (a:select(2,4)-a:select(2,2)+1)):float()
   local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);

   -- intersection over union overlap
   local o = torch.cdiv(inter , (aarea+barea-inter))
   -- set invalid entries to 0 overlap
   o[w:lt(0)] = 0
   o[h:lt(0)] = 0
   return o
end


function utils.intersection(a,b)
   local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
   local x1 = a:select(2,1):clone()
   x1[x1:lt(b[1])] = b[1]
   local y1 = a:select(2,2):clone()
   y1[y1:lt(b[2])] = b[2]
   local x2 = a:select(2,3):clone()
   x2[x2:gt(b[3])] = b[3]
   local y2 = a:select(2,4):clone()
   y2[y2:gt(b[4])] = b[4]

   local w = x2-x1+1;
   local h = y2-y1+1;
   local inter = torch.cmul(w,h):float()
   local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
   (a:select(2,4)-a:select(2,2)+1)):float()
   return torch.cdiv(inter, aarea)
end
--------------------------------------------------------------------------------

function utils.flipBoxes_01(boxes)
   local flipped = boxes:clone()
   flipped:select(2,1):copy(1.0 - boxes:select(2,3))
   flipped:select(2,3):copy(1.0 - boxes:select(2,1))
   return flipped
end

function utils.flipBoxes(boxes, image_width)
   local flipped = boxes:clone()
   flipped:select(2,1):copy( - boxes:select(2,3) + image_width + 1 )
   flipped:select(2,3):copy( - boxes:select(2,1) + image_width + 1 )
   return flipped
end

--------------------------------------------------------------------------------

function utils.merge_table(elements)
   local t = {}
   for i,u in ipairs(elements) do
      for k,v in pairs(u) do
         t[k] = v
      end
   end
   return t
end

-- bbox, tbox: [x1,y1,x2,y2]
local function convertTo(out, bbox, tbox)
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]
      local xtc = (tbox[1] + tbox[3]) * 0.5
      local ytc = (tbox[2] + tbox[4]) * 0.5
      local wt = tbox[3] - tbox[1]
      local ht = tbox[4] - tbox[2]
      out[1] = (xtc - xc) / w
      out[2] = (ytc - yc) / h
      out[3] = math.log(wt / w)
      out[4] = math.log(ht / h)
   else
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]
      local xtc = (tbox[{{},1}] + tbox[{{},3}]) * 0.5
      local ytc = (tbox[{{},2}] + tbox[{{},4}]) * 0.5
      local wt = tbox[{{},3}] - tbox[{{},1}]
      local ht = tbox[{{},4}] - tbox[{{},2}]
      out[{{},1}] = (xtc - xc):cdiv(w)
      out[{{},2}] = (ytc - yc):cdiv(h)
      out[{{},3}] = wt:cdiv(w):log()
      out[{{},4}] = ht:cdiv(h):log()
   end
end

function utils.convertTo(...)
   local arg = {...}
   if #arg == 3 then
      convertTo(...)
   else
      local x = arg[1]:clone()
      convertTo(x, arg[1], arg[2])
      return x
   end
end

function utils.convertFrom(out, bbox, y)
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]

      local xtc = xc + y[1] * w
      local ytc = yc + y[2] * h
      local wt = w * math.exp(y[3])
      local ht = h * math.exp(y[4])

      out[1] = xtc - wt/2
      out[2] = ytc - ht/2
      out[3] = xtc + wt/2
      out[4] = ytc + ht/2
   else
      assert(bbox:size(2) == y:size(2))
      assert(bbox:size(2) == out:size(2))
      assert(bbox:size(1) == y:size(1))
      assert(bbox:size(1) == out:size(1))
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]

      local xtc = torch.addcmul(xc, y[{{},1}], w)
      local ytc = torch.addcmul(yc, y[{{},2}], h)
      local wt = torch.exp(y[{{},3}]):cmul(w)
      local ht = torch.exp(y[{{},4}]):cmul(h)

      out[{{},1}] = xtc - wt * 0.5
      out[{{},2}] = ytc - ht * 0.5
      out[{{},3}] = xtc + wt * 0.5
      out[{{},4}] = ytc + ht * 0.5
   end
end

-- WARNING: DO NOT USE
-- this function is WIP, it doesn't seem to work yet
function utils.setDataParallelN(model, nGPU)
   assert(nGPU)
   assert(nGPU >= 1 and nGPU <= cutorch.getDeviceCount())
   for _,m in ipairs(model:listModules()) do
      if torch.type(m) == 'nn.DataParallelTable' then
         if #m.modules ~= nGPU then
            assert(#m.modules >= 1)
            local inner = m.modules[1]
            inner:float()
            m:__init(m.dimension, m.noGradInput) -- reinitialize
            for i = 1, nGPU do
               cutorch.withDevice(i, function()
                  m:add(inner:clone():cuda(), i)
               end)
            end
         end
      end
   end
   collectgarbage(); collectgarbage();
end

function utils.removeDataParallel(model)
   for _,m in ipairs(model:listModules()) do
      if m.modules then
         for j,inner in ipairs(m.modules) do
            if torch.type(inner) == 'nn.DataParallelTable' then
               assert(#inner.modules >= 1)
               m.modules[j] = inner.modules[1]:float():cuda() -- maybe move to the right GPU
            end
         end
      end
   end
   -- model:float():cuda() -- maybe move to the right GPU
end

-- Deletes entries in modulesToOptState for modules that don't have parameters
-- in the network. This includes modules in DataParallelTable that aren't on
-- the primary GPU.
function utils.cleanupOptim(state)
   local params, gradParams = state.network:parameters()
   local map = {}
   for _,param in ipairs(params) do
      map[param] = true
   end

   local optimizer = state.optimizer
   for module, _ in pairs(optimizer.modulesToOptState) do      
      if torch.type(module.weight) == 'table' then
        local del = false
        for _, par in ipairs(module.weight) do
          if not map[par] then
            del = true
            break
          end
        end
        if del then
          optimizer.modulesToOptState[module] = nil
        end
      else
        if not map[module.weight] and not map[module.bias] then
           optimizer.modulesToOptState[module] = nil
        end
      end
   end
end

function utils.set_NT(model, N, T)
  local old_N, old_T = nil, nil
  -- STMM
  local STMM = model:findModules('nn.STMM')
  if STMM ~= nil then
    for k, v in ipairs(STMM) do
      old_N = old_N or v.N
      old_T = old_T or v.T
      --assert(old_N == v.N and old_T == v.T)
      v.N = N
      v.T = T
    end
  end
  
  -- VidFlip
  local vidFlip = model:findModules('nn.VidFlip')
  if vidFlip ~= nil then
    for k, v in ipairs(vidFlip) do
      old_N = old_N or v.N
      old_T = old_T or v.T
      --assert(old_N == v.N and old_T == v.T)
      v.N = N
      v.T = T
    end
  end
  
  return old_N, old_T
end


function utils.makeProposalPath(proposal_dir, dataset, proposals, set)
   local res = {}
   proposals = stringx.split(proposals, ',')
   for i = 1, #proposals do
      table.insert(res, paths.concat(proposal_dir, dataset, proposals[i], set .. '.t7'))
   end
   return res
end


function utils.saveResults(aboxes, dataset, res_file)

   nClasses = #aboxes
   nImages = #aboxes[1]

   local size = 0
   for class, rc in pairs(aboxes) do
      for i, data in pairs(rc) do
         if data:nElement() > 0 then
            size = size + data:size(1)
         end
      end
   end

   local out = {}
   out.dataset = dataset
   out.images = torch.range(1,nImages):float()
   local det = {}
   out.detections = det
   det.boxes = torch.FloatTensor(size, 4)
   det.scores = torch.FloatTensor(size)
   det.categories = torch.FloatTensor(size)
   det.images = torch.FloatTensor(size)
   local off = 1
   for class = 1, #aboxes do
      for i = 1, #aboxes[class] do
         local data = aboxes[class][i]
         if data:nElement() > 0 then
            det.boxes:narrow(1, off, data:size(1)):copy(data:narrow(2,1,4))
            det.scores:narrow(1, off, data:size(1)):copy(data:select(2,5))
            det.categories:narrow(1, off, data:size(1)):fill(class)
            det.images:narrow(1, off, data:size(1)):fill(i)
            off = off + data:size(1)
         end
      end
   end
   torch.save(res_file, out)
end

-- modified nn.utils
-- accepts different types and numbers
function utils.recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = utils.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resize(t2:size()):copy(t2)
   elseif torch.type(t2) == 'number' then
      t1 = t2
   else
      error("expecting nested tensors or tables. Got "..
      torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function utils.iou_matrix(boxes)
  local n = boxes:size(1)
  local D = boxes.new(n,n):zero()
  for i=1,n do
    local bb = boxes[i]
    D[{i,i}] = 1.0
    for j=i+1,n do
      local bb2 = boxes[j]
      local bi = {math.max(bb[1],bb2[1]), math.max(bb[2],bb2[2]),
                  math.min(bb[3],bb2[3]), math.min(bb[4],bb2[4])}
      local iw = bi[3]-bi[1]+1
      local ih = bi[4]-bi[2]+1
      if iw>0 and ih>0 then
        -- compute overlap as area of intersection / area of union
        local ua = (bb[3]-bb[1]+1)*(bb[4]-bb[2]+1)+
                   (bb2[3]-bb2[1]+1)*(bb2[4]-bb2[2]+1)-iw*ih
        local ov = iw*ih/ua
        D[{i,j}] = ov
        D[{j,i}] = ov
      end
    end
  end
  return D
end


function utils.recursiveCast(dst, src, type)
   if #dst == 0 then
      tnt.utils.table.copy(dst, nn.utils.recursiveType(src, type))
   end
   utils.recursiveCopy(dst, src)
end

-- Expand image into sequences by replicating
function utils.img_to_seq(sample, T)
  local input, target = sample[1], sample[2]
  local N, H, W = input[1]:size(1), input[1]:size(3), input[1]:size(4)
  -- input[1]
  local img = input[1]:view(N, 1, 3, H, W):expand(N, T, 3, H, W):contiguous()
  img = img:view(N*T, 3, H, W)
  input[1] = img
  local rois_coll, target_label_coll, target_coef_coll = {}, {}, {}
  for idx = 1, N do
    local nonzero_idx = torch.nonzero(input[2]:select(2, 1):eq(idx))
    local roi_N = nonzero_idx:nElement()
    if roi_N > 0 then
      nonzero_idx = nonzero_idx:view(-1)
      -- input[2]
      local inc = torch.range(1, T):view(T, 1):expand(T, roi_N):float():contiguous()
      inc:add((idx-1)*T)
      local rois = input[2]:index(1, nonzero_idx)
      rois = rois:view(1, roi_N, 5):expand(T, roi_N, 5):contiguous()
      rois:select(3, 1):copy(inc)
      rois = rois:view(-1, 5)
      table.insert(rois_coll, rois)
      -- target[1] and target[2][1]
      local cur_target_label = target[1]:index(1, nonzero_idx)
      cur_target_label = cur_target_label:view(1, roi_N):expand(T, roi_N):contiguous():view(-1)
      table.insert(target_label_coll, cur_target_label)
      -- target[2][2]
      local cur_target_coef = target[2][2]:index(1, nonzero_idx)
      local D2 = cur_target_coef:size(2)
      cur_target_coef = cur_target_coef:view(1, roi_N, D2):expand(T, roi_N, D2):contiguous():view(T*roi_N, -1)
      table.insert(target_coef_coll, cur_target_coef)
    end
  end
  input[2] = torch.cat(rois_coll, 1)
  target[1] = torch.cat(target_label_coll, 1)
  target[2][1] = target[1]:clone()
  target[2][2] = torch.cat(target_coef_coll, 1)
end

-- another version of nms that returns indexes instead of new boxes
function utils.nms_dense(boxes, overlap)
  local n_boxes = boxes:size(1)

  if n_boxes == 0 then
    return torch.LongTensor()
  end

  -- sort scores in descending order
  assert(boxes:size(2) == 5)
  local vals, I = torch.sort(boxes:select(2,5), 1, true)

  -- sort the boxes
  local boxes_s = boxes:index(1, I):t():contiguous()

  local suppressed = torch.ByteTensor():resize(boxes_s:size(2)):zero()

  local x1 = boxes_s[1]
  local y1 = boxes_s[2]
  local x2 = boxes_s[3]
  local y2 = boxes_s[4]
  local s  = boxes_s[5]

  local area = torch.cmul((x2-x1+1), (y2-y1+1))

  local pick = torch.LongTensor(s:size(1)):zero()

  -- these clones are just for setting the size
  local xx1 = x1:clone()
  local yy1 = x1:clone()
  local xx2 = x1:clone()
  local yy2 = x1:clone()
  local w = x1:clone()
  local h = x1:clone()

  local pickIdx = 1
  for c = 1, n_boxes do
    if suppressed[c] == 0 then
      pick[pickIdx] = I[c]
      pickIdx = pickIdx + 1

      xx1:copy(x1):clamp(x1[c], math.huge)
      yy1:copy(y1):clamp(y1[c], math.huge)
      xx2:copy(x2):clamp(0, x2[c])
      yy2:copy(y2):clamp(0, y2[c])

      w:add(xx2, -1, xx1):add(1):clamp(0, math.huge)
      h:add(yy2, -1, yy1):add(1):clamp(0, math.huge)
      local inter = w
      inter:cmul(h)
      local union = xx1
      union:add(area, -1, inter):add(area[c])
      local ol = h
      torch.cdiv(ol, inter, union)

      suppressed:add(ol:gt(overlap)):clamp(0,1)
    end
  end

  pick = pick[{{1,pickIdx-1}}]
  return pick
end

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k,v in pairs(tbl) do
      -- will skip all DPTs. it also causes stack overflow, idk why
      if torch.typename(v) == 'nn.DataParallelTable' then
         v = v:get(1)
      end
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

utils.deepCopy = deepCopy

function utils.extract_bn_stat(bn) 
  local bn_stat = {}
  bn_stat.eps = bn.eps
  bn_stat.momentum = bn.momentum
  bn_stat.running_mean = bn.running_mean:clone():float()
  bn_stat.running_var = bn.running_var:clone():float()
  return bn_stat
end

function utils.checkpoint(net)
   local net_copy = deepCopy(net):float()
   net_copy:clearState()
   return net_copy
end

function utils.weights_checkpoint(net)
  -- search for batchnorm layers
  --local bn_array = {}
  --if net.STMM then
  --  for _, STMM in ipairs(net.STMM) do
  --    local bn_modules = STMM.net:findModules('cudnn.SpatialBatchNormalization')
  --    for _, bn_module in ipairs(bn_modules) do
  --      local bn_stat = utils.extract_bn_stat(bn_module)
  --      table.insert(bn_array, bn_stat)
  --    end
  --  end
  --end
  
  -- only save weights
  local new_weights = {}
  if torch.type(net) == 'table' then
    for idx, item in ipairs(net) do
      local orig_weights = item:parameters()
      local item_weights = {}
      for i, weights in ipairs(orig_weights) do
        item_weights[i] = weights:clone():float()
      end
      new_weights[idx] = item_weights
    end
  else
    local orig_weights = net:parameters()
    for i, weights in ipairs(orig_weights) do
      new_weights[i] = weights:clone():float()
    end
  end
  
  return new_weights
  --if #bn_array > 0 then
  --  new_weights.bn = bn_array
  --end
  --return deepCopy(net):float():clearState()
end

return utils
