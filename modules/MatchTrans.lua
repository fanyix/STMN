
local MatchTrans, parent = torch.class('nn.MatchTrans', 'nn.Module')
local assemble = require 'assemble'

local MODE = 'POS_AVG'  -- POS_AVG
local AFF_THRESH = 0.0

function MatchTrans:__init(K)
   parent.__init(self)
   self.gradInput = {}
   assert(K % 2==1, 'K must be an odd number.')
   self.K = K
   local pad = (self.K-1)/2
   self.paddor = nn.SpatialZeroPadding(pad,pad,pad,pad)
   self.inv_paddor = nn.SpatialZeroPadding(-pad,-pad,-pad,-pad)
   self.epsilon = 1e-8
end

function MatchTrans:updateOutput(input)
  local H, W, ref_D, N, D
  local cur_map, prev_map, refine_coef, feat
  cur_map, prev_map, feat = table.unpack(input)
  
  N = feat:size(1)
  D = feat:size(2)
  H = feat:size(3)
  W = feat:size(4)
  ref_D = cur_map:size(2)
  
  assert(prev_map:size(3)==H and prev_map:size(4)==W, 
      'Note prev_map should have same spatial size as cur_map.')
  assert(feat:size(3)==H and feat:size(4)==W, 
      'Note feat should have same spatial size as cur_map.')
  
  local pad = (self.K-1)/2
  local cur_prev_affinity = {}
  for idx = 1, N do
    local cur_map_flat = cur_map[{idx, {}, {}, {}}]:view(-1, H*W)
    local prev_map_flat = prev_map[{idx, {}, {}, {}}]:view(-1, H*W)
    local cur_map_flat_norm = torch.norm(cur_map_flat, 2, 1):view(H*W, 1):add(1e-8)
    local prev_map_flat_norm = torch.norm(prev_map_flat, 2, 1):view(1, H*W):add(1e-8)
    local mul_mat = torch.mm(cur_map_flat:t(), prev_map_flat)
    mul_mat:cdiv(cur_map_flat_norm:expandAs(mul_mat))
    mul_mat:cdiv(prev_map_flat_norm:expandAs(mul_mat))
    -- zero-out affinity below threshold
    local mul_mat_flat = mul_mat:view(-1)
    local invalid_idx = mul_mat_flat:le(AFF_THRESH):nonzero()
    
    if invalid_idx:nElement() > 0 then
      invalid_idx = invalid_idx:view(-1)
      mul_mat_flat:indexFill(1, invalid_idx, 0)
    end
    -- collect
    cur_prev_affinity[idx] = mul_mat:view(H,W,H,W)
  end
  
  self.output = self.output or feat.new()
  self.output:resizeAs(feat):zero()
  self.masked_cpa = self.masked_cpa or feat.new() -- masked_cur_prev_affinity
  self.masked_cpa:resize(torch.LongTensor({N,H,W,H,W}):storage()):zero()
  
  for idx = 1, N do
    local inside_loop_timer = torch.Timer()
    
    if MODE == 'POS_AVG' then
      assemble.gpu_assemble(cur_prev_affinity[idx], 
                            feat[{idx, {}, {}, {}}]:view(D, -1), 
                            self.output[{idx, {}, {}, {}}]:view(D, -1), 
                            self.masked_cpa[{idx, {}, {}, {}, {}}], 
                            pad)
    else
      assert(false, 'Unknown MODE option for MatchTrans.')
    end
    
    local inside_loop_time = inside_loop_timer:time().real
  end
  
  return self.output
end

function MatchTrans:updateGradInput(input, gradOutput)
  local timer = torch.Timer()
  local H, W, N, D 
  local cur_map, prev_map, refine_coef, feat
  local pad = (self.K-1)/2
  
  cur_map, prev_map, feat = table.unpack(input)
  self.gradCurMap = self.gradCurMap or cur_map.new()
  self.gradCurMap:resizeAs(cur_map):zero()
  self.gradPrevMap = self.gradPrevMap or prev_map.new()
  self.gradPrevMap:resizeAs(prev_map):zero()
  
  N = feat:size(1)
  D = feat:size(2)
  H = feat:size(3)
  W = feat:size(4)
  self.gradFeat = self.gradFeat or feat.new()
  self.gradFeat:resizeAs(feat):zero()
  
  -- compute self.gradFeat
  for idx = 1, N do
    -- processing
    local masked_cpa = self.masked_cpa[{idx,{},{},{},{}}]:view(H*W, H*W)
    local gradOutput_flat = gradOutput[{idx,{},{},{}}]:view(D, H*W)
    self.gradFeat[{idx,{},{},{}}]:copy(torch.mm(gradOutput_flat, masked_cpa))
  end
  
  self.gradInput = self.gradInput or {}
  self.gradInput[1] = self.gradCurMap
  self.gradInput[2] = self.gradPrevMap
  self.gradInput[3] = self.gradFeat
  
  local time = timer:time().real  
  return self.gradInput
end

