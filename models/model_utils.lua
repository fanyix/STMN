--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local generateGraph = require 'optnet.graphgen'
-- local iterm = require 'iterm'
-- require 'iterm.dot'

local utils = {}

function utils.makeDataParallel(module, nGPU)
  nGPU = nGPU or ((opt and opt.nGPU) or 1)
  if nGPU > 1 then
    --assert(false, 'check carefully since we add rnn after this...')
    local dpt = nn.DataParallelTable(1) -- true?
    local cur_dev = cutorch.getDevice()
    for i = 1, nGPU do
      cutorch.setDevice(i)
      dpt:add(module:clone():cuda(), i)
    end
    cutorch.setDevice(cur_dev)
    return dpt
  else
    return nn.Sequential():add(module)
  end
end


function utils.makeDPParallelTable(module, nGPU)
  if nGPU > 1 then
    local dpt = nn.DPParallelTable()
    local cur_dev = cutorch.getDevice()
    for i = 1, nGPU do
      cutorch.setDevice(i)
      dpt:add(module:clone():cuda(), i)
    end
    cutorch.setDevice(cur_dev)
    return dpt
  else
    return nn.ParallelTable():add(module)
  end
end

-- returns a new Linear layer with less output neurons
function utils.compress(layer, n)
    local W = layer.weight
    local U,S,V = torch.svd(W:t():float())
    local new = nn.Linear(W:size(2), n):cuda()
    new.weight:t():copy(U:narrow(2,1,n) * torch.diag(S:narrow(1,1,n)) * V:narrow(1,1,n):narrow(2,1,n))
    new.bias:zero()
    return new
end

-- returns a Sequential of 2 Linear layers, one biasless with U*diag(S) and one
-- with V and original bias. L is the number of components to keep.
function utils.SVDlinear(layer, L)
  local W = layer.weight:double()
  local b = layer.bias:double()

  local K, N = W:size(1), W:size(2)

  local U, S, V = torch.svd(W:t(), 'A')

  local US = U:narrow(2,1,L) * torch.diag(S:narrow(1,1,L))
  local Vt = V:narrow(2,1,L)

  local L1 = nn.LinearNB(N, L)
  L1.weight:copy(US:t())

  local L2 = nn.Linear(L, K)
  L2.weight:copy(Vt)
  L2.bias:copy(b)

  return nn.Sequential():add(L1):add(L2)
end


function utils.testSurgery(input, f, net, ...)
   local output1 = net:forward(input)
   if torch.type(output1) == 'table' then
    output1 = output1[1]:clone()
   end
   f(net,...)
   local output2 = net:forward(input)
   if torch.type(output2) == 'table' then
    output2 = output2[1]:clone()
   end
   print((output1 - output2):abs():max())
   assert((output1 - output2):abs():max() < 1e-5)
end

function utils.conv_to_STMM(model, conv53, WU_mode)
  local STMMSet = model:findModules('nn.STMM')
  local mem_dim = STMMSet[1].M
  local feat_dim = STMMSet[1].D
  local dim_count = 0
  for idx, STMM in ipairs(STMMSet) do
    local conv53_W = conv53[1][{{dim_count+1,dim_count+mem_dim},{},{},{}}]
    local conv53_b = conv53[2][{{dim_count+1,dim_count+mem_dim}}]
    dim_count = dim_count + mem_dim
    
    local conv_to_be_modified
    if WU_mode == 'mem_W' then
      conv_to_be_modified = {STMM.conv_w}
    elseif WU_mode == 'W' then
      conv_to_be_modified = {STMM.conv_w, STMM.conv_z_w, STMM.conv_r_w}
      --if STMM.elemwise_mask then
      --  conv_to_be_modified = {STMM.conv_w, STMM.conv_z_w, STMM.conv_r_w}
      --else
      --  conv_to_be_modified = {STMM.conv_w}
      --end
    elseif WU_mode == 'WU' then
      assert(feat_dim == mem_dim, 'This mode only works when feat_dim == mem_dim.')
      conv_to_be_modified = {STMM.conv_w, STMM.conv_z_w, STMM.conv_r_w, 
                              STMM.conv_u, STMM.conv_z_u, STMM.conv_r_u}
    else
      assert(false, 'Unknown mode.')
    end

    for _, cur_conv in ipairs(conv_to_be_modified) do
      local weight = cur_conv.weight
      local bias = cur_conv.bias
      weight:copy(conv53_W)
      bias:copy(conv53_b)
    end
  end
  assert(dim_count==conv53[1]:size(1), 'The conv53 is not used fully.')
  
  return model
end

function utils.removeDropouts(net)
  net:replace(function(x)
    return torch.typename(x):find'nn.Dropout' and nn.Identity() or x
  end)
end


function utils.disableFeatureBackprop(features, maxLayer)
  local noBackpropModules = nn.Sequential()
  for i = 1,maxLayer do
    noBackpropModules:add(features.modules[1])
    features:remove(1)
  end
  features:insert(nn.NoBackprop(noBackpropModules):cuda(), 1)
end

function utils.classBBoxMotionLinear(N, N2)
  local class_linear = nn.Linear(N, opt.num_classes):cuda()
  class_linear.weight:normal(0,0.01)
  class_linear.bias:zero()

  local bbox_linear = nn.Linear(N2 or N, opt.num_classes*4):cuda()
  bbox_linear.weight:normal(0,0.001)
  bbox_linear.bias:zero()
  
  local mot_linear = nn.Linear(N2 or N, opt.num_classes*4):cuda()
  mot_linear.weight:normal(0,0.001)
  mot_linear.bias:zero()

  if N2 then
    return nn.ParallelTable():add(class_linear):add(bbox_linear):add(mot_linear):cuda()
  else
    return nn.ConcatTable():add(class_linear):add(bbox_linear):add(mot_linear):cuda()
  end
end

function utils.bboxLinear(N)
  local bbox_linear = nn.Linear(N, opt.num_classes * 4):cuda()
  bbox_linear.weight:normal(0,0.001)
  bbox_linear.bias:zero()
  return bbox_linear
end

function utils.classAndBBoxLinear(N, N2)
  local class_linear = nn.Linear(N,opt and opt.num_classes or 21):cuda()
  class_linear.weight:normal(0,0.01)
  class_linear.bias:zero()

  local bbox_linear = nn.Linear(N2 or N,(opt and opt.num_classes or 21) * 4):cuda()
  bbox_linear.weight:normal(0,0.001)
  bbox_linear.bias:zero()

  if N2 then
    return nn.ParallelTable():add(class_linear):add(bbox_linear):cuda()
  else
    return nn.ConcatTable():add(class_linear):add(bbox_linear):cuda()
  end
end


function utils.fill_bn_stat(bn_module, bn_stat) 
  bn_module.eps = bn_stat.eps
  bn_module.momentum = bn_stat.momentum
  assert(bn_module.running_mean:nElement() == bn_stat.running_mean:nElement())
  assert(bn_module.running_var:nElement() == bn_stat.running_var:nElement())
  bn_module.running_mean:copy(bn_stat.running_mean)
  bn_module.running_var:copy(bn_stat.running_var)
end


function utils.copyBN(src_model, tgt_model)
  local module_set = {'cudnn.SpatialBatchNormalization', 
                      'nn.SpatialBatchNormalization', 
                      'nn.BatchNormalization'}
  for module_idx, module_name in ipairs(module_set) do
    local src_BNs = src_model:findModules(module_name)
    local tgt_BNs = tgt_model:findModules(module_name)
    assert(#src_BNs == #tgt_BNs, string.format(
      'Source and target has different number of module %s', module_name))
    for idx, tgt_BN in ipairs(tgt_BNs) do
      local src_BN = src_BNs[idx]      
      --src_BN.running_mean:zero()
      --tgt_BN.running_mean:zero()
      --src_BN.running_var:fill(1)
      --tgt_BN.running_var:fill(1)
      tgt_BN.running_mean:copy(src_BN.running_mean)
      tgt_BN.running_var:copy(src_BN.running_var)
      if src_BN.save_mean and src_BN.save_mean:nElement() > 0 then
        tgt_BN.save_mean:copy(src_BN.save_mean)
      end
      if src_BN.save_std and src_BN.save_std:nElement() > 0 then
        tgt_BN.save_std:copy(src_BN.save_std)
      end
      if src_BN.weight and src_BN.weight:nElement() > 0 then
        tgt_BN.weight:copy(src_BN.weight)
      end
      if src_BN.bias and src_BN.bias:nElement() > 0 then
        tgt_BN.bias:copy(src_BN.bias)
      end
    end
  end
end


function utils.convert_model_RGB2Flow(model)
  local ru = paths.dofile 'refiner_utils.lua'
  local nobp = model:findModules('nn.NoBackprop')[1]
  local seq = nobp:get(1)
  local rgb_input_conv = seq:get(1)
  local rgb_weight = rgb_input_conv.weight
  local i, o, h, w = rgb_weight:size(1), rgb_weight:size(2), rgb_weight:size(3), rgb_weight:size(4)
  local rgb_bias = rgb_input_conv.bias
  local flow_input_conv = cudnn.SpatialConvolution(2, 64, 3, 3, 1, 1, 1, 1)
  local flow_weight = torch.mean(rgb_weight, 2):expand(i, 2, h, w)
  flow_weight:mul(o / 2)
  flow_input_conv.weight:copy(flow_weight)
  flow_input_conv.bias:copy(rgb_bias)
  seq:remove(1)
  seq:insert(flow_input_conv, 1)
  ru.syncParameters(model)
  return model
end


function utils.loadModelWeights(src_params, tgt_model, mode)
  -- mode    full: load weights for all layers
  --         vgg_conv: only load vgg conv parts (1-26 layers)
  
  mode = mode or 'full'
  local tgt_params = tgt_model:parameters()
  
  if mode == 'full' then
    -- replace the vgg weights
    --assert(#src_params == #tgt_params)
    if #src_params ~= #tgt_params then
      local loop = true
      while loop do
        io.write("source and target param blobs do not match, continue (y/n)? ")
        io.flush()
        local answer=io.read()
        if answer == 'y' then
          loop = false 
        elseif answer == 'n' then
          assert(false, 'Source and target param blobs do not match. Terminated.')
        end
      end
    end
    for idx = 1, #tgt_params do
      assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
      tgt_params[idx]:copy(src_params[idx])
    end
  
  elseif mode == 'resnet101_bottleneck' then
    local bottleneck_module_idx = 189
    for idx = 1, #src_params do
      if idx < bottleneck_module_idx then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      else
        assert(src_params[idx]:nElement() == tgt_params[idx+2]:nElement())
        tgt_params[idx+2]:copy(src_params[idx])
      end
    end
  elseif mode == 'vgg_conv' then
    local vgg_conv_layernum = 26
    assert(#src_params >= vgg_conv_layernum)
    for idx = 1, vgg_conv_layernum do
      assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
      tgt_params[idx]:copy(src_params[idx])
    end
  elseif mode == 'vgg_conv52_STMM' or mode == 'vgg_conv52_STMM_W' or 
          mode == 'vgg_conv52_STMM_WU' then
    local vgg_conv_layernum = 24
    assert(#src_params >= vgg_conv_layernum)
    for idx = 1, vgg_conv_layernum do
      assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
      tgt_params[idx]:copy(src_params[idx])
    end
    
    local last_conv = {src_params[vgg_conv_layernum+1], src_params[vgg_conv_layernum+2]}
    if mode == 'vgg_conv52_STMM_W' then
      -- copy-paste conv5_3 to STMM 
      tgt_model = utils.conv_to_STMM(tgt_model, last_conv, 'W')
    elseif mode == 'vgg_conv52_STMM_WU' then
      -- copy-paste conv5_3 to STMM 
      tgt_model = utils.conv_to_STMM(tgt_model, last_conv, 'WU')
    end
    
  elseif mode == 'resnet50_conv52_STMM' or mode == 'resnet50_conv52_STMM_W' or 
          mode == 'resnet50_conv52_STMM_WU' then
    local resnet50_conv_layernum = 86
    assert(#src_params >= resnet50_conv_layernum)
    for idx = 1, resnet50_conv_layernum do
      assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
      tgt_params[idx]:copy(src_params[idx])
    end
    local last_conv = {src_params[resnet50_conv_layernum+1], src_params[resnet50_conv_layernum+2]}
    if mode == 'resnet50_conv52_STMM_W' then
      -- copy-paste conv5_3 to STMM
      tgt_model = utils.conv_to_STMM(tgt_model, last_conv, 'W')
    elseif mode == 'resnet50_conv52_STMM_WU' then
      -- copy-paste conv5_3 to STMM
      tgt_model = utils.conv_to_STMM(tgt_model, last_conv, 'WU')
    end
    -- fill in the fc layers
    local tgt_start_idx = 89
    local src_start_idx = 89
    local src_end_idx = 108
    for src_idx = src_start_idx, src_end_idx do
      local tgt_idx = tgt_start_idx - src_start_idx + src_idx
      assert(src_params[src_idx]:nElement() == tgt_params[tgt_idx]:nElement())
      tgt_params[tgt_idx]:copy(src_params[src_idx])
    end
  elseif mode == 'resnet101_STMM_W' then
    local bottleneck_module_idx = 189
    for idx = 1, #src_params do
      if idx < bottleneck_module_idx then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      elseif idx > bottleneck_module_idx + 1 then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      end
    end
    tgt_model = utils.conv_to_STMM(tgt_model, {src_params[bottleneck_module_idx], 
                src_params[bottleneck_module_idx + 1]}, 'W')
  elseif mode == 'resnet101_STMM_mem_W' then
    local bottleneck_module_idx = 189
    for idx = 1, #src_params do
      if idx < bottleneck_module_idx then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      elseif idx > bottleneck_module_idx + 1 then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      end
    end
    tgt_model = utils.conv_to_STMM(tgt_model, {src_params[bottleneck_module_idx], 
                src_params[bottleneck_module_idx + 1]}, 'mem_W')
  elseif mode == 'rfcn_STMM_W' then
    local bottleneck_module_idx = 213
    for idx = 1, #src_params do
      if idx < bottleneck_module_idx then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      elseif idx > bottleneck_module_idx + 1 then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      end
    end
    tgt_model = utils.conv_to_STMM(tgt_model, {src_params[bottleneck_module_idx], 
                src_params[bottleneck_module_idx + 1]}, 'W')
  elseif mode == 'rfcn_STMM_mem_W' then
    local bottleneck_module_idx = 213
    for idx = 1, #src_params do
      if idx < bottleneck_module_idx then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      elseif idx > bottleneck_module_idx + 1 then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      end
    end
    tgt_model = utils.conv_to_STMM(tgt_model, {src_params[bottleneck_module_idx], 
                src_params[bottleneck_module_idx + 1]}, 'mem_W')
  elseif mode == 'rfcn_STMM_mem_W_warmStart' then
    local bottleneck_module_idx = 213
    for idx = 1, #src_params do
      if idx < bottleneck_module_idx then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      elseif idx > bottleneck_module_idx + 1 then
        assert(src_params[idx]:nElement() == tgt_params[idx]:nElement())
        tgt_params[idx]:copy(src_params[idx])
      end
    end
    tgt_model = utils.conv_to_STMM(tgt_model, {src_params[bottleneck_module_idx], 
                src_params[bottleneck_module_idx + 1]}, 'mem_W')
    local STMMSet = tgt_model:findModules('nn.STMM')
    for idx, STMM in ipairs(STMMSet) do
      STMM.conv_u.weight:mul(0.1)
      STMM.conv_z_w.bias:add(2)
    end
  else
    assert(false, 'Unknown model loading mode.')
  end
  
  -- fill in bn stats, if there is 
  --if src_params.bn and tgt_model.STMM then
  --  local bn_stats = src_params.bn
  --  local bn_stats_counter = 1
  --  assert(#bn_stats == #tgt_model.STMM * 2, 
  --    'The number of bn struct should be twice of the number of STMM.')
  --  for _, STMM in ipairs(tgt_model.STMM) do
  --    local bn_modules = STMM.net:findModules('cudnn.SpatialBatchNormalization')
  --    for _, bn_module in ipairs(bn_modules) do
  --      utils.fill_bn_stat(bn_module, bn_stats[bn_stats_counter])
  --      bn_stats_counter = bn_stats_counter + 1
  --    end
  --  end
  --end
end


function utils.testModel(model)
  input_size = model.input_size or 224
  print(model)
  model:training()
  local batchSz = opt.seq_per_batch * opt.timestep_per_batch
  local boxes = torch.Tensor(opt.nGPU, 5)
  for i = 1, opt.nGPU do
    boxes[i]:copy(torch.Tensor({1,1,1,100,100}))
  end
  local images = torch.rand(batchSz,3,input_size,input_size):cuda()
  local input = {images,boxes:cuda()}  
  local output = model:forward(input)
  --local graphOpts = {
  --  displayProps =  {color='dodgerblue'},
  --}
  --graph.dot(generateGraph(model, input, graphOpts), 'VID', 'VID')
  --iterm.dot(generateGraph(model, input), opt and opt.save_folder..'/graph.pdf' or paths.tmpname()..'.pdf')
  print{output}
  print{model:backward(input,output)}
end

-- used in AlexNet and VGG models trained by Ross
function utils.RossTransformer()
  return fbcoco.ImageTransformer({102.9801,115.9465,122.7717}, nil, 255, {3,2,1})
end

function utils.VGGFlowTransformer()
  -- note 56 is the std of imagenet images
  --return fbcoco.ImageTransformer({127.6079, 127.3543}, {17.2075 / 56, 11.1054 / 56}, 255)
  return fbcoco.ImageTransformer({127.6079, 127.3543}, nil, 255)
end

-- used in ResNet and facebook inceptions
function utils.ImagenetTransformer()
  return fbcoco.ImageTransformer(
  { -- mean
    0.48462227599918,
    0.45624044862054,
    0.40588363755159,
  },
  { -- std
    0.22889466674951,
    0.22446679341259,
    0.22495548344775,
  })
end

function utils.normalizeBBoxRegr(model, meanstd)
  if #model:findModules('nn.BBoxNorm') == 0 then
    -- normalize the bbox regression
    local regression_layer = model:get(#model.modules):get(2)
    if torch.type(regression_layer) == 'nn.Sequential' then
      regression_layer = regression_layer:get(#regression_layer.modules)
    end
    assert(torch.type(regression_layer) == 'nn.Linear')

    local mean_hat = torch.repeatTensor(meanstd.mean,1,opt.num_classes):cuda()
    local sigma_hat = torch.repeatTensor(meanstd.std,1,opt.num_classes):cuda()

    regression_layer.weight:cdiv(sigma_hat:t():expandAs(regression_layer.weight))
    regression_layer.bias:add(-mean_hat):cdiv(sigma_hat)

    utils.addBBoxNorm(model, meanstd)
  end
end

function utils.addBBoxNorm(model, meanstd)
  
  -- This is a hack for STMM_RESNET
  if model.bottom_stack then
    model = model.bottom_stack
  end

  if #model:findModules('nn.BBoxNorm') == 0 then
    model:add(
        nn.ParallelTable()
          :add(nn.Identity())
          :add(nn.BBoxNorm(meanstd.mean, meanstd.std)):cuda()
    )
  end
end

function utils.vggSetPhase2(model)
  assert(model.phase == 1)
  local dpt = model.modules[1].modules[1]
  for i = 1, #dpt.modules do
    assert(torch.type(dpt.modules[i]) == 'nn.NoBackprop')
    dpt.modules[i] = dpt.modules[i].modules[1]
    utils.disableFeatureBackprop(dpt.modules[i], 10)
  end
  model.phase = phase
  print("Switched model to phase 2")
  print(model)
end

function utils.vggSetPhase2_outer(model)
  assert(model.phase == 1)
  model.modules[1].modules[1] = model.modules[1].modules[1].modules[1]
  local dpt = model.modules[1].modules[1]
  for i = 1, #dpt.modules do
    utils.disableFeatureBackprop(dpt.modules[i], 10)
  end
  model.phase = phase
  print("Switched model to phase 2")
  print(model)
end

function utils.conv345Combine(isNormalized, useConv3, useConv4, initCopyConv5)
  local totalFeat = 0

  local function make1PoolingLayer(idx, nFeat, spatialScale, normFactor)
    local pool1 = nn.Sequential()
      :add(nn.ParallelTable():add(nn.SelectTable(idx)):add(nn.Identity()))
      :add(inn.ROIPooling(7,7,spatialScale))
    if isNormalized then
      pool1:add(nn.View(-1, nFeat*7*7))
        :add(nn.Normalize(2))
        :add(nn.Contiguous())
        :add(nn.View(-1, nFeat, 7, 7))
    else
      pool1:add(nn.MulConstant(normFactor))
    end
    totalFeat = totalFeat + nFeat
    return pool1
  end

  local pooling_layer = nn.ConcatTable()
  pooling_layer:add(make1PoolingLayer(1, 512, 1/16, 1)) -- conv5
  if useConv4 then
    pooling_layer:add(make1PoolingLayer(2, 512, 1/8, 1/30)) -- conv4
  end
  if useConv3 then
    pooling_layer:add(make1PoolingLayer(3, 256, 1/4, 1/200)) -- conv3
  end
  local pooling_join = nn.Sequential()
    :add(pooling_layer)
    :add(nn.JoinTable(2))
  if isNormalized then
    pooling_join:add(nn.MulConstant(1000))
  end
  local conv_mix = cudnn.SpatialConvolution(totalFeat, 512, 1, 1, 1, 1)
  if initCopyConv5 then
    conv_mix.weight:zero()
    conv_mix.weight:narrow(2, 1, 512):copy(torch.eye(512)) -- initialize to just copy conv5
  end
  pooling_join:add(conv_mix)
  pooling_join:add(nn.View(-1):setNumInputDims(3))

  return pooling_join
end

-- workaround for bytecode incompat functions
function utils.safe_unpack(self)
   if self.unpack and self.model then
      return self:unpack()
   else
      local model = self.model
      for k,v in ipairs(model:listModules()) do
         if v.weight and not v.gradWeight then
            v.gradWeight = v.weight:clone()
            v.gradBias = v.bias:clone()
         end
      end
      return model
   end
end

function utils.load(path)
  local data = torch.load(path)
  return data.unpack and data:unpack() or data
end


function utils.freezeWeights(model)
  local cur_modules = nil
  local vgg_conv_layernum = 26
  cur_modules = model.modules
  for idx = 1, vgg_conv_layernum do
    assert(cur_modules[idx].accGradParameters and cur_modules[idx].updateParameters)
    cur_modules[idx].accGradParameters = function() end
    cur_modules[idx].updateParameters = function() end
  end
  
  return model
end

return utils
