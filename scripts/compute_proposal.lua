
-- This script is used to compute deep/sharp mask upon images
package.path = package.path .. ';../?.lua'

require 'deepmask.SharpMask'
require 'deepmask.SpatialSymmetricPadding'
require 'deepmask.InferSharpMask'
require 'image'

local model_utils = require 'models.model_utils'
local utils = require 'utils'
local coco = require 'coco'
local myutils = require 'myutils'

local cmd = torch.CmdLine()
cmd:option('-np', 1000, 'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', true, 'use DeepMask version of SharpMask')
cmd:option('-img','./multipathnet/deepmask/data/testImage.jpg' ,'path/to/test/image')
cmd:option('-maxsize', 600, 'resize image dimension')
cmd:option('-savefile', 'sharpmask', 'filename for saving the results')
local config = cmd:parse(arg)
local this_file_dir = paths.dirname(paths.thisfile(nil))

local dataset_dir = '/home/fanyix/code/VID/UCF101/'
local cache_dir = paths.concat(this_file_dir, '../../data/')
local sharpmask_path = paths.concat(cache_dir, 'models/sharpmask.t7')
local prop_filename = paths.concat(dataset_dir, string.format('exp/proposals/test/%s.t7', config.savefile))
local root_dir = paths.concat(dataset_dir, 'data')
local lock_dir = paths.concat(dataset_dir, 'exp/locks')
print(string.format('Save to %s', prop_filename))


local sharpmask = torch.load(sharpmask_path).model
sharpmask:inference(config.np)

------------------- DeepMask Settings --------------------

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end
print(scales)
local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = sharpmask,
  dm = config.dm,
}
local counter, finished_video_count = 1, 0
local timer
local accum_time = 0
local prop = {}

------------------- Run DeepMask --------------------

--------- BELOW IS FOR IMAGENET VID DATA ---------

--local dir_list = myutils.dir(root_dir)
---- iterate over all sub-directories
--for dir_idx, dir in ipairs(dir_list) do
--  local video_tag = dir
--  local file_list = myutils.dir(paths.concat(root_dir, dir))
--  
--  -- check lock
--  local lock_filename = paths.concat(lock_dir, string.format('%s', dir))
--  paths.mkdir(paths.dirname(lock_filename))
--  
--  if not paths.filep(lock_filename) then
--    
--    -- write lock
--    local fd = io.open(lock_filename, 'w')
--    fd:write('lock')
--    fd:close()
--  
--    local cur_vid_prop = {images = {}, scores = {}, boxes = {}, images_idx = {}}
--  
--    for file_idx, file in ipairs(file_list) do
--      -- timing
--      if cutorch then cutorch.synchronize() end
--      timer = torch.Timer()
--
--      local cur_file = paths.concat(root_dir, dir, file)
--      local img = image.load(cur_file)
--      img = image.scale(img, config.maxsize)
--      local h,w = img:size(2),img:size(3)
--      
--      local forward_timer = torch.Timer()
--      infer:forward(img)
--      local forward_time = forward_timer:time().real
--      
--      local postproc_timer = torch.Timer()
--      local masks, scores = infer:getTopProps(.2,h,w)
--      local postproc_time = postproc_timer:time().real
--      
--      local org_timer = torch.Timer()
--      scores = scores[{{}, 1}]:clone()
--      local Rs = coco.MaskApi.encode(masks)
--      local bboxes = coco.MaskApi.toBbox(Rs)
--      bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
--      
--      -- normalize box coordinates
--      local scaler = bboxes.new({w, h, w, h}):view(1, 4)
--      bboxes:cdiv(scaler:expandAs(bboxes))
--      
--      table.insert(cur_vid_prop.images, file)
--      table.insert(cur_vid_prop.scores, scores:float())
--      table.insert(cur_vid_prop.boxes, bboxes:float())
--      table.insert(cur_vid_prop.images_idx, tonumber(paths.basename(file, 'JPEG')))
--      local org_time = org_timer:time().real
--      
--      -- timing
--      if cutorch then cutorch.synchronize() end
--      local time = timer:time().real
--      accum_time = accum_time + time
--      
--      -- logging
--      print(string.format('%g/%g  |  %g/%g  |  time=%g (%g,%g,%g)  |  accu time=%g', 
--            dir_idx, #dir_list, file_idx, #file_list, accum_time / counter, 
--            forward_time, postproc_time, org_time, accum_time))
--      
--      -- visualization
--      --local qtwidget = require 'qtwidget'
--      --local win = qtwidget.newwindow(img:size(3), img:size(2))  
--      --coco.MaskApi.drawMasks(img, masks, 10)
--      --image.display({image = img, win = win})
--      --io.flush()
--      --local answer=io.read()      
--      --win:close()
--      
--      -- incremnt counter
--      counter = counter + 1
--    end
--    
--    -- sort by frame
--    local frame_idx = torch.IntTensor(cur_vid_prop.images_idx)
--    local sort_val, sort_idx = torch.sort(frame_idx)
--    local sorted_cur_vid_prop = {images = {}, scores = {}, boxes = {}}
--    for _, ii in ipairs(sort_idx:totable()) do
--      table.insert(sorted_cur_vid_prop.images, cur_vid_prop.images[ii])
--      table.insert(sorted_cur_vid_prop.scores, cur_vid_prop.scores[ii])
--      table.insert(sorted_cur_vid_prop.boxes, cur_vid_prop.boxes[ii])
--      --table.insert(sorted_cur_vid_prop.images_idx, cur_vid_prop.images_idx[ii])
--    end
--    
--    -- collect 
--    sorted_cur_vid_prop.scores = torch.cat(sorted_cur_vid_prop.scores, 2)
--    sorted_cur_vid_prop.boxes = torch.cat(sorted_cur_vid_prop.boxes, 3):permute(3,1,2):contiguous()
--    prop[video_tag] = sorted_cur_vid_prop
--    
--    -- incremnt finished_video_count
--    finished_video_count = finished_video_count + 1
--    
--    -- saving
--    if finished_video_count % 20 == 0 then
--      paths.mkdir(paths.dirname(prop_filename))
--      torch.save(prop_filename, prop)
--    end
--    
--    -- collect memory garbage
--    collectgarbage()
--  end
--end
--
---- save finally
--paths.mkdir(paths.dirname(prop_filename))
--torch.save(prop_filename, prop)
--print(string.format('Saved to %s', prop_filename))
--print('Done.')


--------- BELOW IS FOR IMAGENET DET DATA ---------

--prop = {images={}, scores={}, boxes={}}
--local dir_list = myutils.dir(root_dir)
---- iterate over all sub-directories
--for dir_idx, dir in ipairs(dir_list) do
--  local file_list = myutils.dir(paths.concat(root_dir, dir))
--  
--  for file_idx, file in ipairs(file_list) do
--    -- check lock
--    local tag = dir .. '/' .. file
--    local lock_tag = dir .. '_' .. file
--    local lock_filename = paths.concat(lock_dir, string.format('%s', lock_tag))
--    paths.mkdir(paths.dirname(lock_filename))
--    
--    if not paths.filep(lock_filename) then
--      
--      -- write lock
--      local fd = io.open(lock_filename, 'w')
--      fd:write('lock')
--      fd:close()
--      
--      -- timing
--      if cutorch then cutorch.synchronize() end
--      timer = torch.Timer()
--
--      local cur_file = paths.concat(root_dir, dir, file)
--      local img = image.load(cur_file)
--      assert(img:dim() == 3)
--      if img:size(1) == 1 then
--        -- grey image
--        img = img:expand(3, img:size(2), img:size(3)):contiguous()
--      end
--      img = image.scale(img, config.maxsize)
--      local h,w = img:size(2),img:size(3)
--      
--      local forward_timer = torch.Timer()
--      infer:forward(img)
--      local forward_time = forward_timer:time().real
--      
--      local postproc_timer = torch.Timer()
--      local masks, scores = infer:getTopProps(.2,h,w)
--      local postproc_time = postproc_timer:time().real
--      
--      local org_timer = torch.Timer()
--      scores = scores[{{}, 1}]:clone()
--      local Rs = coco.MaskApi.encode(masks)
--      local bboxes = coco.MaskApi.toBbox(Rs)
--      bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
--      
--      -- normalize box coordinates
--      local scaler = bboxes.new({w, h, w, h}):view(1, 4)
--      bboxes:cdiv(scaler:expandAs(bboxes))
--      
--      -- collect results
--      prop.images[counter] = tag
--      prop.scores[counter] = scores
--      prop.boxes[counter] = bboxes
--      
--      -- timing
--      local org_time = org_timer:time().real
--      if cutorch then cutorch.synchronize() end
--      local time = timer:time().real
--      accum_time = accum_time + time
--      
--      -- logging
--      print(string.format('%g/%g  |  %g/%g  |  time=%g (%g,%g,%g)  |  accu time=%g', 
--            dir_idx, #dir_list, file_idx, #file_list, accum_time / counter, 
--            forward_time, postproc_time, org_time, accum_time))
--      
--      -- visualization
--      --local qtwidget = require 'qtwidget'
--      --local win = qtwidget.newwindow(img:size(3), img:size(2))  
--      --coco.MaskApi.drawMasks(img, masks, 10)
--      --image.display({image = img, win = win})
--      --io.flush()
--      --local answer=io.read()      
--      --win:close()
--      
--      -- incremnt counter
--      counter = counter + 1
--      
--      -- saving
--      if counter % 1000 == 0 then
--        paths.mkdir(paths.dirname(prop_filename))
--        torch.save(prop_filename, prop)
--      end
--      
--      -- collect memory garbage
--      collectgarbage()
--    end
--  end
--end

-- save finally
--paths.mkdir(paths.dirname(prop_filename))
--torch.save(prop_filename, prop)
--print(string.format('Saved to %s', prop_filename))
--print('Done.')

--------- BELOW IS FOR JHMDB DATA ---------

---- load annotation for which we compute proposals
--local anno_filename = '/home/SSD3/fanyi-data/jhmdb/exp/annotation/test.t7'
--local anno = torch.load(anno_filename)
--local total_vid_num = #myutils.keys(anno)
--local vid_count = 0
--local ov_thresh = 0.5
--local total, tp = 0, 0
--
--for vid_name, vid in pairs(anno) do
--  -- check lock
--  local lock_filename = paths.concat(lock_dir, string.format('%s', vid_name))
--  paths.mkdir(paths.dirname(lock_filename))
--  
--  if not paths.filep(lock_filename) then
--    -- write lock
--    local fd = io.open(lock_filename, 'w')
--    fd:write('lock')
--    fd:close()
--    
--    local cur_vid_prop = {images = {}, scores = {}, boxes = {}, images_idx = {}}
--    local prop_filename = paths.concat(dataset_dir, string.format('exp/proposals/%s.t7', vid_name))
--    
--    for file_idx, file in ipairs(vid.im_list) do
--      -- timing
--      if cutorch then cutorch.synchronize() end
--      timer = torch.Timer()
--
--      local cur_file = paths.concat(root_dir, vid_name, file)
--      local img = image.load(cur_file)
--      img = image.scale(img, config.maxsize)
--      local h,w = img:size(2),img:size(3)
--      
--      local forward_timer = torch.Timer()
--      infer:forward(img)
--      local forward_time = forward_timer:time().real
--      
--      local postproc_timer = torch.Timer()
--      local masks, scores = infer:getTopProps(.2,h,w)
--      local postproc_time = postproc_timer:time().real
--      
--      local org_timer = torch.Timer()
--      scores = scores[{{}, 1}]:clone()
--      local Rs = coco.MaskApi.encode(masks)
--      local bboxes = coco.MaskApi.toBbox(Rs)
--      bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
--      
--      -- normalize box coordinates
--      local scaler = bboxes.new({w, h, w, h}):view(1, 4)
--      bboxes:cdiv(scaler:expandAs(bboxes))
--      
--      table.insert(cur_vid_prop.images, file)
--      table.insert(cur_vid_prop.scores, scores:float())
--      table.insert(cur_vid_prop.boxes, bboxes:float())
--      table.insert(cur_vid_prop.images_idx, tonumber(paths.basename(file, 'png')))
--      local org_time = org_timer:time().real
--      
--      -- compute the overlap with GT if possible
--      if file_idx <= vid.obj[1].boxes:size(1) then
--        local ov = utils.boxoverlap_01(cur_vid_prop.boxes[file_idx], vid.obj[1].boxes[{file_idx, {3, 6}}])
--        local maxov = torch.max(ov)
--        if maxov > ov_thresh then
--          tp = tp + 1
--        end
--        total = total + 1
--      end
--      
--      -- timing
--      if cutorch then cutorch.synchronize() end
--      local time = timer:time().real
--      accum_time = accum_time + time
--      
--      -- logging
--      print(string.format('%g/%g  |  %g/%g  |  time=%g (%g,%g,%g)  |  accu time=%g  |  recall=%g', 
--            vid_count, total_vid_num, file_idx, #vid.im_list, accum_time / counter, 
--            forward_time, postproc_time, org_time, accum_time, tp/total))
--      
--      -- visualization
--      --local qtwidget = require 'qtwidget'
--      --local win = qtwidget.newwindow(img:size(3), img:size(2))  
--      --coco.MaskApi.drawMasks(img, masks, 10)
--      --image.display({image = img, win = win})
--      --io.flush()
--      --local answer=io.read()      
--      --win:close()
--      
--      -- incremnt counter
--      counter = counter + 1
--    end
--    
--    -- sort by frame
--    local frame_idx = torch.IntTensor(cur_vid_prop.images_idx)
--    local sort_val, sort_idx = torch.sort(frame_idx)
--    local sorted_cur_vid_prop = {images = {}, scores = {}, boxes = {}}
--    for _, ii in ipairs(sort_idx:totable()) do
--      table.insert(sorted_cur_vid_prop.images, cur_vid_prop.images[ii])
--      table.insert(sorted_cur_vid_prop.scores, cur_vid_prop.scores[ii])
--      table.insert(sorted_cur_vid_prop.boxes, cur_vid_prop.boxes[ii])
--      --table.insert(sorted_cur_vid_prop.images_idx, cur_vid_prop.images_idx[ii])
--    end
--    
--    -- collect 
--    sorted_cur_vid_prop.scores = torch.cat(sorted_cur_vid_prop.scores, 2)
--    sorted_cur_vid_prop.boxes = torch.cat(sorted_cur_vid_prop.boxes, 3):permute(3,1,2):contiguous()
--    
--    -- saving
--    paths.mkdir(paths.dirname(prop_filename))
--    torch.save(prop_filename, sorted_cur_vid_prop)
--    
--    -- increment video count
--    vid_count = vid_count + 1
--    
--    -- collect memory garbage
--    collectgarbage()
--  end
--end

--------- BELOW IS FOR JHMDB DATA ---------

-- load annotation for which we compute proposals
local anno_filename = '/home/fanyix/code/VID/UCF101/exp/annotation/train.t7'
local anno = torch.load(anno_filename)
local total_vid_num = #myutils.keys(anno)
local vid_count = 0
local ov_thresh = 0.5
local total, tp = 0, 0

for vid_name, vid in pairs(anno) do
  -- check lock
  local lock_filename = paths.concat(lock_dir, string.format('%s', vid_name))
  paths.mkdir(paths.dirname(lock_filename))
  
  if not paths.filep(lock_filename) then
    -- write lock
    local fd = io.open(lock_filename, 'w')
    fd:write('lock')
    fd:close()
    
    local cur_vid_prop = {images = {}, scores = {}, boxes = {}, images_idx = {}}
    local prop_filename = paths.concat(dataset_dir, string.format('exp/proposals/%s.t7', vid_name))
    
    for file_idx, file in ipairs(vid.im_list) do
      -- timing
      if cutorch then cutorch.synchronize() end
      timer = torch.Timer()

      local cur_file = paths.concat(root_dir, vid_name, file)
      local img = image.load(cur_file)
      img = image.scale(img, config.maxsize)
      local h,w = img:size(2),img:size(3)
      
      local forward_timer = torch.Timer()
      infer:forward(img)
      local forward_time = forward_timer:time().real
      
      local postproc_timer = torch.Timer()
      local masks, scores = infer:getTopProps(.2,h,w)
      local postproc_time = postproc_timer:time().real
      
      local org_timer = torch.Timer()
      scores = scores[{{}, 1}]:clone()
      local Rs = coco.MaskApi.encode(masks)
      local bboxes = coco.MaskApi.toBbox(Rs)
      bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
      
      -- normalize box coordinates
      local scaler = bboxes.new({w, h, w, h}):view(1, 4)
      bboxes:cdiv(scaler:expandAs(bboxes))
      
      table.insert(cur_vid_prop.images, file)
      table.insert(cur_vid_prop.scores, scores:float())
      table.insert(cur_vid_prop.boxes, bboxes:float())
      table.insert(cur_vid_prop.images_idx, tonumber(paths.basename(file, 'jpg')))
      local org_time = org_timer:time().real
      
      -- compute the overlap with GT if possible
      if file_idx <= vid.obj[1].boxes:size(1) then
        local ov = utils.boxoverlap_01(cur_vid_prop.boxes[file_idx], vid.obj[1].boxes[{file_idx, {3, 6}}])
        local maxov = torch.max(ov)
        if maxov > ov_thresh then
          tp = tp + 1
        end
        total = total + 1
      end
      
      -- timing
      if cutorch then cutorch.synchronize() end
      local time = timer:time().real
      accum_time = accum_time + time
      
      -- logging
      print(string.format('%g/%g  |  %g/%g  |  time=%g (%g,%g,%g)  |  accu time=%g  |  recall=%g', 
            vid_count, total_vid_num, file_idx, #vid.im_list, accum_time / counter, 
            forward_time, postproc_time, org_time, accum_time, tp/total))
      
      -- visualization
      --local qtwidget = require 'qtwidget'
      --local win = qtwidget.newwindow(img:size(3), img:size(2))  
      --coco.MaskApi.drawMasks(img, masks, 10)
      --image.display({image = img, win = win})
      --io.flush()
      --local answer=io.read()      
      --win:close()
      
      -- incremnt counter
      counter = counter + 1
    end
    
    -- sort by frame
    local frame_idx = torch.IntTensor(cur_vid_prop.images_idx)
    local sort_val, sort_idx = torch.sort(frame_idx)
    local sorted_cur_vid_prop = {images = {}, scores = {}, boxes = {}}
    for _, ii in ipairs(sort_idx:totable()) do
      table.insert(sorted_cur_vid_prop.images, cur_vid_prop.images[ii])
      table.insert(sorted_cur_vid_prop.scores, cur_vid_prop.scores[ii])
      table.insert(sorted_cur_vid_prop.boxes, cur_vid_prop.boxes[ii])
      --table.insert(sorted_cur_vid_prop.images_idx, cur_vid_prop.images_idx[ii])
    end
    
    -- collect 
    sorted_cur_vid_prop.scores = torch.cat(sorted_cur_vid_prop.scores, 2)
    sorted_cur_vid_prop.boxes = torch.cat(sorted_cur_vid_prop.boxes, 3):permute(3,1,2):contiguous()
    
    -- saving
    paths.mkdir(paths.dirname(prop_filename))
    torch.save(prop_filename, sorted_cur_vid_prop)
    
    -- increment video count
    vid_count = vid_count + 1
    
    -- collect memory garbage
    collectgarbage()
  end
end


