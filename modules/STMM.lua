
require 'nngraph'
require 'cudnn'
require 'modules.AdaLinearScale'
require 'modules.MatchTrans'

local STMM, parent = torch.class('nn.STMM','nn.Module')
local ELEMENT_MASK = true  -- Controls whether having an elementwise z and r, or single layer ones
local MEM_ALIGN = true
local std_multiplier = 3.0
local mem_rewgt = 10.0
local mem_align_K = 5
local heavy_lifter = cudnn
local manualSeed = nil

function STMM:__init(N, T, D, M, MULT)
  parent.__init(self)
  self.N = N
  self.T = T
  self.D = D
  self.M = M
  self.t = 1
  self.elemwise_mask = ELEMENT_MASK
  self.cell_constructed_flag = false
  self.mult_after_w = MULT or 1
  
  if manualSeed ~= nil then
    math.randomseed(manualSeed)
    cutorch.manualSeedAll(manualSeed)
    torch.manualSeed(manualSeed)
  end
  
  -- init three conv modules
  local conv_z_w, conv_z_u, conv_r_w, conv_r_u
  local conv_w = heavy_lifter.SpatialConvolution(D, M, 3, 3, 1, 1, 1, 1)
  local conv_u = heavy_lifter.SpatialConvolution(M, M, 3, 3, 1, 1, 1, 1)
  if ELEMENT_MASK then
    conv_z_w = heavy_lifter.SpatialConvolution(D, M, 3, 3, 1, 1, 1, 1)
    conv_z_u = heavy_lifter.SpatialConvolution(M, M, 3, 3, 1, 1, 1, 1)
    conv_r_w = heavy_lifter.SpatialConvolution(D, M, 3, 3, 1, 1, 1, 1)
    conv_r_u = heavy_lifter.SpatialConvolution(M, M, 3, 3, 1, 1, 1, 1)
  else
    conv_z_w = heavy_lifter.SpatialConvolution(D, 1, 3, 3, 1, 1, 1, 1)
    conv_z_u = heavy_lifter.SpatialConvolution(M, 1, 3, 3, 1, 1, 1, 1)
    conv_r_w = heavy_lifter.SpatialConvolution(D, 1, 3, 3, 1, 1, 1, 1)
    conv_r_u = heavy_lifter.SpatialConvolution(M, 1, 3, 3, 1, 1, 1, 1)
  end
  
  -- cancel the bias
  conv_u:noBias()
  conv_z_u:noBias()
  conv_r_u:noBias()
  
  -- init 
  self.concat = nn.JoinTable(1)
  local feat_input = nn.Identity()()
  local prev_feat_input = nn.Identity()()
  local mem_input = nn.Identity()()
  local output_node, z_node, r_node = self:construct_STMMCell(conv_w, conv_u, conv_z_w, conv_z_u, 
                      conv_r_w, conv_r_u, feat_input, prev_feat_input, mem_input)
  self.net = nn.gModule({feat_input, prev_feat_input, mem_input}, {output_node, z_node, r_node})
  -- link params of these three convs to the params of this module
  local params, grad_params = self.net:getParameters()
  self.weight = {params}
  self.gradWeight = {grad_params}
end

function STMM:parameters()
   return self.weight, self.gradWeight
end

function STMM:construct_STMMCell(conv_w, conv_u, conv_z_w, conv_z_u, 
                      conv_r_w, conv_r_u, feat_node, prev_feat_node, mem_node)
  assert(self.cell_constructed_flag==false, 'Cannot construct the cell twice.')
  self.cell_constructed_flag = true
    
  if MEM_ALIGN then
    mem_node = nn.MatchTrans(mem_align_K)({feat_node, prev_feat_node, mem_node})
  else
    feat_node = nn.CAddTable()({feat_node, nn.MulConstant(0)(prev_feat_node)})
  end
  
  -- init local convs
  self.conv_z_w = conv_z_w:clone('weight', 'bias', 'gradWeight', 'gradBias')
  self.conv_r_w = conv_r_w:clone('weight', 'bias', 'gradWeight', 'gradBias')
  self.conv_w = conv_w:clone('weight', 'bias', 'gradWeight', 'gradBias')
  self.conv_z_u = conv_z_u:clone('weight', 'bias', 'gradWeight', 'gradBias')
  self.conv_r_u = conv_r_u:clone('weight', 'bias', 'gradWeight', 'gradBias')
  self.conv_u = conv_u:clone('weight', 'bias', 'gradWeight', 'gradBias')
  
  local z = nn.AdaLinearScale(std_multiplier)(
      heavy_lifter.ReLU()(nn.CAddTable()({nn.MulConstant(self.mult_after_w)(
      self.conv_z_w(feat_node)), self.conv_z_u(mem_node)})))
  local r = nn.AdaLinearScale(std_multiplier)(
      heavy_lifter.ReLU()(nn.CAddTable()({nn.MulConstant(self.mult_after_w)(
      self.conv_r_w(feat_node)), self.conv_r_u(mem_node)})))
  
  if not ELEMENT_MASK then
    z = nn.Squeeze(3)(nn.Replicate(self.M, 2)(z))
    r = nn.Squeeze(3)(nn.Replicate(self.M, 2)(r))
  end
  local rMem = nn.CMulTable()({r, mem_node})
  
  local candMem = heavy_lifter.ReLU()(nn.CAddTable()({nn.MulConstant(
                    self.mult_after_w)(self.conv_w(feat_node)), self.conv_u(rMem)}))
  
  local zCandMem = nn.CMulTable()({z, candMem})
  local oneMinusZ = nn.AddConstant(1)(nn.MulConstant(-1)(z))
  local oneMinusZMem = nn.CMulTable()({oneMinusZ, mem_node})
  local newMem = nn.CAddTable()({zCandMem, oneMinusZMem})
  self.t = self.t + 1
  
  return newMem, z, r
end

function STMM:updateOutput(input)
  self.recompute_backward = true
  
  local D, H, W
  local N, T, M = self.N, self.T, self.M
  local input_5d, mem0
  
  if torch.type(input) == self:type() then
    D, H, W = input:size(2), input:size(3), input:size(4)  
    input_5d = input:view(N, T, D, H, W)
    mem0 = torch.FloatTensor(N, M, H, W):zero():type(self:type())
  else
    D, H, W = input[1]:size(2), input[1]:size(3), input[1]:size(4)
    input_5d = input[1]:view(N, T, D, H, W)
    mem0 = input[2]
  end
  
  -- execute forward loop
  local mem_input = mem0
  self.mem_input = {}
  self.mem_output = {}
  self.z_output = {}
  self.r_output = {}
  for t = 1, T do
    local feature_input = input_5d[{{}, t, {}, {}, {}}]
    local prev_feature_input
    if t == 1 then
      prev_feature_input = input_5d[{{}, t, {}, {}, {}}]
    else
      prev_feature_input = input_5d[{{}, t-1, {}, {}, {}}]
    end
    
    self.mem_input[t] = mem_input
    local tmp_out = self.net:forward({feature_input, prev_feature_input, mem_input})
    mem_input = tmp_out[1]:clone()
    self.mem_output[t] = mem_input
    self.z_output[t] = tmp_out[2]:clone()
    self.r_output[t] = tmp_out[3]:clone()
  end
  -- get output
  local output = self.concat:forward(self.mem_output)
  self.output = self.output or output.new(N*T, M, H, W)
  self.output:resizeAs(output)
  self.output:copy(output:view(T, N, M, H, W):transpose(1, 2):contiguous():view(N*T, M, H, W))
  return self.output
end

function STMM:backward(input,gradOutput)
  self.recompute_backward = false
  
  local D, H, W
  local N, T, M = self.N, self.T, self.M
  local input_5d, mem0
  
  if torch.type(input) == self:type() then
    D, H, W = input:size(2), input:size(3), input:size(4)  
    input_5d = input:view(N, T, D, H, W)
    mem0 = torch.FloatTensor(N, M, H, W):zero():type(self:type())
  else
    D, H, W = input[1]:size(2), input[1]:size(3), input[1]:size(4)
    input_5d = input[1]:view(N, T, D, H, W)
    mem0 = input[2]
  end
  gradOutput = gradOutput:view(N, T, M, H, W)
  
  -- backward loop
  self.z_gradOutput = self.z_gradOutput or self.z_output[1].new()
  self.r_gradOutput = self.r_gradOutput or self.r_output[1].new()
  self.z_gradOutput:resizeAs(self.z_output[1]):zero()
  self.r_gradOutput:resizeAs(self.r_output[1]):zero()
  self.mem_gradInput = self.mem_gradInput or gradOutput.new(N, M, H, W)
  self.mem_gradInput:resize(N, M, H, W):zero()
  self.feat_gradInput = self.feat_gradInput or gradOutput.new(N*T, D, H, W)
  self.feat_gradInput:resize(N*T, D, H, W):zero()
  local feature_grad = self.feat_gradInput:view(N, T, D, H, W)
  for t = T, 1, -1 do
    local feature_input = input_5d[{{}, t, {}, {}, {}}]
    self.mem_gradInput:add(gradOutput[{{}, t, {}, {}, {}}])
    local prev_feature_input    
    if t == 1 then
      prev_feature_input = input_5d[{{}, t, {}, {}, {}}]
    else
      prev_feature_input = input_5d[{{}, t-1, {}, {}, {}}]
    end
    -- we need to do this to make sure all intermediate results inside self.net is correct
    self.net:forward({feature_input, prev_feature_input, self.mem_input[t]})
    local grad_input = self.net:backward({feature_input, prev_feature_input, 
                        self.mem_input[t]}, {self.mem_gradInput, self.z_gradOutput, self.r_gradOutput})
    feature_grad[{{}, t, {}, {}, {}}]:copy(grad_input[1])
    self.mem_gradInput:copy(grad_input[3])
  end
  
  
  if torch.type(input) == self:type() then
    self.gradInput = self.feat_gradInput
  else
    self.gradInput = {self.feat_gradInput, self.mem_gradInput}
  end
  return self.gradInput
end

function STMM:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function STMM:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end
