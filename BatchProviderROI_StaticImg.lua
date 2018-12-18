--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local BatchProviderROI_StaticImg, parent = torch.class('fbcoco.BatchProviderROI_StaticImg', 'fbcoco.BatchProviderBase_StaticImg')
local utils = paths.dofile'utils.lua'
local tablex = require'pl.tablex'

function BatchProviderROI_StaticImg:__init(dataset, transformer, fg_threshold, bg_threshold, opt)
   assert(transformer,'must provide transformer!')

   self.dataset = dataset

   self.batch_size = 128
   self.fg_fraction = 0.25

   self.fg_threshold = fg_threshold
   self.bg_threshold = bg_threshold

   self.imgs_per_batch = opt.images_per_batch
   self.scale = opt.scale
   self.max_size = opt.max_size
   self.image_transformer = transformer

   self.scale_jitter    = opt.scale_jitter or 0    -- uniformly jitter the scale by this frac
   self.aspect_jitter   = opt.aspect_jitter or 0   -- uniformly jitter the scale by this frac
   self.crop_likelihood = opt.crop_likelihood or 0 -- likelihood of doing a random crop (in each dimension, independently)
   self.crop_attempts = 10                     -- number of attempts to try to find a valid crop
   self.crop_min_frac = 0.7                             -- a crop must preserve at least this fraction of the iamge
   
   if opt.brightness_var and opt.contrast_var and opt.saturation_var and opt.lighting_var then
     self.color_jitter = fbcoco.ColorTransformer(opt.brightness_var, 
                          opt.contrast_var, opt.saturation_var, opt.lighting_var)
   else
     self.color_jitter = nil
   end
   
end

-- Prepare foreground / background rois for one image
-- there is a check if self.bboxes has a table prepared for this image already
-- because we prepare the rois during training to save time on loading
function BatchProviderROI_StaticImg:setupOne(i)
   local rec = self.dataset:attachProposals(i)

   local bf = rec.overlap:ge(self.fg_threshold):nonzero()
   local bg = rec.overlap:ge(self.bg_threshold[1]):cmul(
   rec.overlap:lt(self.bg_threshold[2])):nonzero()
   return {
      [0] = self.takeSubset(rec, bg, i, true),
      [1] = self.takeSubset(rec, bf, i, false)
   }
end

-- Calculate rois and supporting data for the first 1000 images
-- to compute mean/var for bbox regresion
function BatchProviderROI_StaticImg:setupData()
   local regression_values = {}
   local subset_size = math.min(self.dataset:size(), 1000)
   for i = 1, subset_size do
      local v = self:setupOne(i)[1]
      if v then
         table.insert(regression_values, utils.convertTo(v.rois, v.gtboxes))
      end
   end
   regression_values = torch.FloatTensor():cat(regression_values,1)

   self.bbox_regr = {
      mean = regression_values:mean(1),
      std = regression_values:std(1)
   }
   return self.bbox_regr
end

-- sample until find a valid combination of bg/fg boxes
function BatchProviderROI_StaticImg:permuteIdx()
   local boxes, img_idx = {}, {}
   for i=1,self.imgs_per_batch do
      local curr_idx
      local bboxes = {}
      while not bboxes[0] or not bboxes[1] do
         curr_idx = torch.random(self.dataset:size())
         tablex.update(bboxes, self:setupOne(curr_idx))
      end
      table.insert(boxes, bboxes)
      table.insert(img_idx, curr_idx)
   end
   local do_flip = torch.FloatTensor(self.imgs_per_batch):random(0,1)
   return torch.IntTensor(img_idx), boxes, do_flip
end

function BatchProviderROI_StaticImg:selectBBoxes(boxes, im_scales, im_sizes, do_flip)
   local rois = {}
   local labels = {}
   local gtboxes = {}
   for im,v in ipairs(boxes) do
      local flip = do_flip[im] == 1

      local bg = self.selectBBoxesOne(v[0],self.bg_num_each,im_scales[im],im_sizes[im],flip)
      local fg = self.selectBBoxesOne(v[1],self.fg_num_each,im_scales[im],im_sizes[im],flip)

      local imrois = torch.FloatTensor():cat(bg.rois, fg.rois, 1)
      imrois = torch.FloatTensor(imrois:size(1),1):fill(im):cat(imrois, 2)
      local imgtboxes = torch.FloatTensor():cat(bg.gtboxes, fg.gtboxes, 1)
      local imlabels = torch.IntTensor():cat(bg.labels, fg.labels, 1)

      table.insert(rois, imrois)
      table.insert(gtboxes, imgtboxes)
      table.insert(labels, imlabels)
   end
   gtboxes = torch.FloatTensor():cat(gtboxes,1)
   rois = torch.FloatTensor():cat(rois,1)
   labels = torch.IntTensor():cat(labels,1)
   return rois, labels, gtboxes
end


function BatchProviderROI_StaticImg:sample()
   collectgarbage()
   self.fg_num_each = self.fg_fraction * self.batch_size
   self.bg_num_each = self.batch_size - self.fg_num_each

   local img_idx, boxes, do_flip = self:permuteIdx()
   local images, im_scales, im_sizes = self:getImages(img_idx, do_flip)
   local rois, labels, gtboxes = self:selectBBoxes(boxes, im_scales, im_sizes, do_flip)

   local bboxregr_vals = torch.FloatTensor(rois:size(1), 4*(self.dataset:getNumClasses() + 1)):zero()

   for i,label in ipairs(labels:totable()) do
      if label > 1 then
         local out = bboxregr_vals[i]:narrow(1,(label-1)*4 + 1,4)
         utils.convertTo(out, rois[i]:narrow(1,2,4), gtboxes[i])
         out:add(-1,self.bbox_regr.mean):cdiv(self.bbox_regr.std)
      end
   end

    -- DIAGONOZE DISPLAY
    local DEBUG = false
    if DEBUG then
      self.cat_name_to_id = {}
      self.cat_id_to_name = {}
      local category_list = {'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 
              'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 
              'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 
              'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra'
      }
      self.class_num = #category_list
      for cat_id, cat_name in ipairs(category_list) do
        self.cat_name_to_id[cat_name] = cat_id + 1 -- +1 because we leave 1 to background class
        self.cat_id_to_name[cat_id + 1] = cat_name
      end
      local qtwidget = require 'qtwidget'
      
      local win = qtwidget.newwindow(images:size(4), images:size(3))  
      local reg_win = qtwidget.newwindow(images:size(4), images:size(3))  
      for debug_idx = 1, rois:size(1) do
        local debug_label = labels[debug_idx]
        --if debug_label > 1 then
        if true then
          local debug_img_idx = rois[debug_idx][1]
          -- raw box
          local out = rois[{debug_idx, {2, 5}}]:clone()
          local x1 = out[1]
          local y1 = out[2]
          local x2 = out[3]
          local y2 = out[4]
          image.display({image = images[{debug_img_idx, {}, {}, {}}], win = win})
          win:fill()
          win:rectangle(x1, y1, x2-x1+1, y2-y1+1)
          win:stroke()
          -- reg box
          local reg_out = bboxregr_vals[debug_idx]:narrow(1,(debug_label-1)*4 + 1,4):clone()
          reg_out:cmul(self.bbox_regr.std):add(1,self.bbox_regr.mean)
          local raw_box = rois[{debug_idx, {2, 5}}]
          utils.convertFrom(reg_out, raw_box, reg_out)
          local x1 = reg_out[1]
          local y1 = reg_out[2]
          local x2 = reg_out[3]
          local y2 = reg_out[4]
          image.display({image = images[{debug_img_idx, {}, {}, {}}], win = reg_win})
          reg_win:fill()
          reg_win:rectangle(x1, y1, x2-x1+1, y2-y1+1)
          reg_win:stroke()
          print(string.format('cat:%s', self.cat_id_to_name[debug_label]))
          print('-----')
        end
      end
    end
    
    local batches = {images, rois}
    local targets = {labels, {labels, bboxregr_vals}, g_donkey_idx}
    return batches, targets
end
