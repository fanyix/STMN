
local myutils = require 'myutils'
local utils = paths.dofile'utils.lua'
local tnt = require 'torchnet'

------------------------- AUX FUNCTIONS -------------------------

local function alignPropToGT(cur_prop, cur_gt, K)
  local fg_threshold = opt.fg_threshold
  local record = {}
  local regression_values = {}
  
  for ii = 1, K do
    -- sample one ground truth track 
    local sample_obj_idx = torch.random(1, #cur_gt.obj)
    local gt = cur_gt.obj[sample_obj_idx]
    
    -- sample one frame from the track
    local sample_frame_idx = torch.random(1, gt.boxes:size(1))
    
    -- register
    local skip = false
    if record[sample_obj_idx] == nil then
      record[sample_obj_idx] = {}
      record[sample_obj_idx][sample_frame_idx] = true
    else
      if record[sample_obj_idx][sample_frame_idx] == nil then
        record[sample_obj_idx][sample_frame_idx] = true
      else
        -- visited before
        skip = true
      end
    end
    
    if not skip then
      -- get gt and proposals for this frame
      local frame_id = gt.boxes[{sample_frame_idx, 2}]
      local gt_box = utils.calibrate_box_01(gt.boxes[{sample_frame_idx, {3, 6}}]:clone())
      local prop_box
      if torch.type(cur_prop.boxes) == 'table' then
        prop_box = utils.calibrate_box_01(cur_prop.boxes[frame_id]:clone())
      elseif torch.isTensor(cur_prop.boxes) then
        prop_box = utils.calibrate_box_01(cur_prop.boxes[{frame_id, {}, {}}]:clone())
      else
        assert(false, 'Unknown proposal format.')
      end
      
      -- sample 
      local o = utils.boxoverlap_01(prop_box, gt_box)
      local valid_idx = torch.nonzero(o:ge(fg_threshold))
      if valid_idx:nElement() > 0 then
        local align_prop_box = prop_box:index(1, valid_idx:view(-1))
        local align_gt_box = gt_box:view(1, 4):expandAs(align_prop_box)
        table.insert(regression_values, utils.convertTo(align_prop_box, align_gt_box))
      end
    end
  end
  -- flatten
  if #regression_values > 0 then
    regression_values = torch.cat(regression_values, 1)
  else
    regression_values = torch.FloatTensor(0, 4)
  end
  return regression_values
end


local function computeRegStats(anno, prop_dir, K)
  local video_names = myutils.keys(anno)
  local anno_N = #video_names
  local sample_per_video = 1
  local vid_K = math.ceil(K / sample_per_video) or anno_N
  vid_K = math.min(vid_K, anno_N)
  local perm = torch.randperm(anno_N)
  local regression_values = {}
  for i = 1, vid_K do
    local key = video_names[perm[i]]
    local cur_prop = torch.load(paths.concat(prop_dir, string.format('%s.t7', key)))
    --local cur_prop = prop[key]
    local cur_gt = anno[key]
    local reg_coef = alignPropToGT(cur_prop, cur_gt, sample_per_video)
    if reg_coef:nElement() > 0 then
      table.insert(regression_values, reg_coef)
    end
  end
  regression_values = torch.FloatTensor():cat(regression_values,1)
  local bbox_regr = {
    mean = regression_values:mean(1),
    std = regression_values:std(1)
  }
  return bbox_regr
end


local function createVideoTrainLoader(opt, anno)
  local transformer = torch.load(opt.transformer)
  local bp_opt = tnt.utils.table.clone(opt)
  bp_opt.bg_threshold = {opt.bg_threshold_min, opt.bg_threshold_max}
  local bp = fbcoco.BatchProviderVID(anno, transformer, bp_opt)
  bp.class_specific = opt.train_class_specific
  return bp
end

----------------------- LOAD DATASET INFO -------------------------
-- load the annotation file
local anno = torch.load(opt.anno_file)
local local_opt = tnt.utils.table.clone(opt)

------------------------- CREATE REGRESSION STATS -------------------------
local bbox_regr = computeRegStats(anno, local_opt.prop_dir, 1000)
g_mean_std = bbox_regr

------------------------- CREATE LOADER -------------------------
local loader = createVideoTrainLoader(local_opt, anno)
loader.bbox_regr = bbox_regr

-- automatically compute epochSize
local_opt.epochSize = math.ceil(#myutils.keys(loader.anno) / loader.batch_N)
opt.epochSize = local_opt.epochSize


local function getIterator()
  local dataset = tnt.ListDataset{
                    list = torch.range(1,local_opt.epochSize):long(),
                    load = function(idx)
                       return {loader:sample()}
                    end,
                  }
  local iterator = tnt.DatasetIterator(dataset)
  return iterator
end


local function getParallelIterator()
   return tnt.ParallelDatasetIterator{
      nthread = local_opt.nDonkeys,
      init = function(idx)
         local this_file_dir = paths.dirname(paths.thisfile(nil))
         package.path = package.path .. string.format(';%s/?.lua', this_file_dir)
         require 'torchnet'
         require 'fbcoco'
         torch.manualSeed(local_opt.manualSeed + idx)
         --require 'donkey'
         --g_donkey_idx = idx
      end,
      closure = function()         
         local loader = createVideoTrainLoader(local_opt, anno)
         loader.bbox_regr = bbox_regr
         return tnt.ListDataset{
            list = torch.range(1,local_opt.epochSize):long(),
            load = function(idx)
               return {loader:sample()}
            end,
         }
      end,
   }
end


return {getIterator, getParallelIterator, createVideoTrainLoader, loader}
