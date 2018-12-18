--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'inn'
require 'cudnn'
require 'fbcoco'
require 'rfcn'
require 'modules.STMM'
require 'modules.VidFlip'
require 'modules.DbTunnel'
require 'modules.BilinearRoiPooling'
local utils = paths.dofile'model_utils.lua'
local mu = paths.dofile'../myutils.lua'

-- architecture options
local STMM_MODULE = nn.STMM
local RESNET_BOTTLENECK_MAG = 0.05  -- 0.05
local RESNET_MODEL_TYPE = 'RESNET101'
local RESNET_FREEZE_LAYER = 5  -- freeze up to layer 5, or 7 if freeze feature entirely

local model, transformer, flow_transformer
local fthis = paths.dirname(paths.thisfile(nil))

inn.utils = require 'inn.utils'

local function loadResNet(model_path)
   local net = torch.load(model_path)
   net:cuda():evaluate()
   local features = nn.Sequential()
   for i=1,7 do features:add(net:get(i)) end
   local input = torch.randn(1,3,224,224):cuda()
   utils.testSurgery(input, utils.disableFeatureBackprop, features, 5)
   utils.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
   utils.testSurgery(input, inn.utils.BNtoFixed, features, false)
   utils.testSurgery(input, inn.utils.BNtoFixed, net, false)
   
   if RESNET_FREEZE_LAYER == 7 then
    utils.testSurgery(input, utils.disableFeatureBackprop, features, 3)
   end
   local classifier = nn.Sequential()
   for i=8,10 do classifier:add(net:get(i)) end
   local output_dim = classifier.output:size(2)
   
   local conv = classifier:get(1):get(1):get(1):get(1):get(4)
   local new_conv = nn.SpatialConvolution(conv.nInputPlane, 
       conv.nOutputPlane, conv.kW, conv.kH, 1, 1, conv.padW, conv.padH)
   new_conv.weight:copy(conv.weight)
   new_conv.bias:copy(conv.bias)
   classifier:get(1):get(1):get(1):get(1):remove(4)
   classifier:get(1):get(1):get(1):get(1):insert(new_conv, 4)
  
   local conv = classifier:get(1):get(1):get(1):get(2):get(1)
   local new_conv = nn.SpatialConvolution(conv.nInputPlane, 
       conv.nOutputPlane, conv.kW, conv.kH, 1, 1, conv.padW, conv.padH)
   new_conv.weight:copy(conv.weight)
   new_conv.bias:copy(conv.bias)
   classifier:get(1):get(1):get(1):get(2):remove(1)
   classifier:get(1):get(1):get(1):get(2):insert(new_conv, 1)
       local conv = classifier:get(1):get(2):get(1):get(1):get(4)
   local dilated_conv = nn.SpatialDilatedConvolution(conv.nInputPlane, 
       conv.nOutputPlane, conv.kW, conv.kH, conv.stride[1], conv.stride[2], conv.padW+1, conv.padH+1, 2, 2)
   dilated_conv.weight:copy(conv.weight)
   dilated_conv.bias:copy(conv.bias)
   classifier:get(1):get(2):get(1):get(1):remove(4)
   classifier:get(1):get(2):get(1):get(1):insert(dilated_conv, 4)
  
   local conv = classifier:get(1):get(3):get(1):get(1):get(4)
   local dilated_conv = nn.SpatialDilatedConvolution(conv.nInputPlane, 
       conv.nOutputPlane, conv.kW, conv.kH, conv.stride[1], conv.stride[2], conv.padW+1, conv.padH+1, 2, 2)
   dilated_conv.weight:copy(conv.weight)
   dilated_conv.bias:copy(conv.bias)
   classifier:get(1):get(3):get(1):get(1):remove(4)
   classifier:get(1):get(3):get(1):get(1):insert(dilated_conv, 4)
  
   classifier:remove(3)
   classifier:remove(2)
   local conv = nn.SpatialDilatedConvolution(2048, 512, 3, 3, 1, 1, 6, 6, 6, 6)
   classifier:add(conv)
   classifier:add(cudnn.ReLU())
   
   local grid_size = 7     
   local rfcn_head = nn.Sequential()
                      :add(nn.ConcatTable()
                            :add(nn.Sequential()
                                  :add(nn.ParallelTable()
                                      :add(nn.SpatialConvolution(512, grid_size*grid_size*opt.num_classes, 1, 1, 1, 1, 0, 0))
                                      :add(nn.Identity())
                                  )
                                  :add(rfcn.PSROIPooling(1/16, grid_size, opt.num_classes))
                                  :add(cudnn.SpatialAveragePooling(grid_size, grid_size, 1, 1, 0, 0))
                                  :add(nn.View(-1, opt.num_classes))
                            )
                            :add(nn.Sequential()
                                  :add(nn.ParallelTable()
                                      :add(nn.SpatialConvolution(512, grid_size*grid_size*4, 1, 1, 1, 1, 0, 0))
                                      :add(nn.Identity())
                                  )
                                  :add(rfcn.PSROIPooling(1/16, grid_size, 4))
                                  :add(cudnn.SpatialAveragePooling(grid_size, grid_size, 1, 1, 0, 0))
                                  :add(nn.View(-1, 4))
                                  :add(nn.Replicate(opt.num_classes, 2))
                                  :add(nn.Contiguous())
                                  :add(nn.View(-1, 4*opt.num_classes))
                            )
                      )
   collectgarbage()
   
    local model
    if opt.model == 'stmn' then
      local comp_N, comp_T, comp_M = opt.seq_per_batch, opt.timestep_per_batch, 1024
      local layer = nn.Sequential()
      local branches = nn.ConcatTable()
      local STMM_1 = STMM_MODULE(comp_N, comp_T, 1024, comp_M/2, 1.0/RESNET_BOTTLENECK_MAG)
      local STMM_2 = STMM_MODULE(comp_N, comp_T, 1024, comp_M/2, 1.0/RESNET_BOTTLENECK_MAG)
      branches:add(nn.Sequential():add(STMM_1))
      branches:add(
                  nn.Sequential()
                  :add(nn.VidFlip(comp_N, comp_T))
                  :add(STMM_2)
                  :add(nn.VidFlip(comp_N, comp_T))
      )
      layer:add(branches):add(nn.JoinTable(2))
      classifier:add(nn.SpatialConvolution(512, 1024, 1, 1, 1, 1, 0, 0))
                :add(cudnn.ReLU())        
      features = nn.Sequential():add(features):add(classifier)
      features = utils.makeDataParallel(features)
      local bottom = nn.Sequential()
                       :add(nn.ParallelTable()
                              :add(nn.SpatialConvolution(1024, 512, 1, 1, 1, 1, 0, 0))
                              :add(nn.Identity())
                       )
                       :add(rfcn_head)
      model = nn.Sequential()
        :add(nn.ParallelTable()
          :add(nn.Sequential()
                :add(features)
                :add(layer)
          )
         :add(nn.Identity())
        )
        :add(bottom)
      model.conv_stack = features
      model.STMM_stack = layer
      model.post_stack = bottom
    elseif opt.model == 'rfcn' then
      local embed_network = nn.Sequential()
                             :add(nn.SpatialConvolution(512, 1024, 1, 1, 1, 1, 0, 0))
                             :add(cudnn.ReLU())
                             :add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
                             :add(cudnn.ReLU())
                             :add(nn.SpatialConvolution(1024, 512, 1, 1, 1, 1, 0, 0))
      classifier:add(embed_network)
      -- add classification and regression heads
      model = nn.Sequential()
        :add(nn.ParallelTable()
          :add(utils.makeDataParallel(
                nn.Sequential()
                  :add(features)
                  :add(classifier))
          )
         :add(nn.Identity())
        )
        :add(rfcn_head)
    else
      assert(false, 'Unknown model type.')
    end
   
   model:cuda()
   utils.testModel(model)
   return model
end

local resnet_model_file = '../../dataset/ImageNetVID/models/resnet-101.t7'

if fthis and fthis ~= "" then
  resnet_model_file = paths.concat(fthis, resnet_model_file)
end
model = loadResNet(resnet_model_file)

transformer = paths.concat(fthis, '..', '..', 'dataset', 'ImageNetVID', 'models', 'resnet_transformer.t7')
if not mu.file_exists(transformer) then
  local T = utils.ImagenetTransformer()
  torch.save(transformer, T)
end

model:cuda()

utils.testModel(model)

return {model, transformer, flow_transformer}
