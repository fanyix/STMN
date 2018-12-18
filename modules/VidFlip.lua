require 'nngraph'

local VidFlip, parent = torch.class('nn.VidFlip','nn.Module')

function VidFlip:__init(N, T)
  parent.__init(self)
  self.N = N
  self.T = T
end

function VidFlip:updateOutput(input)
  local K, D, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  local N, T = self.N, self.T
  assert(K == N * T, 'Total number of image does not match.')
  local input_5d = input:view(N, T, D, H, W)
  self.output = (self.output:nElement() > 0 and self.output) or input:clone():zero()
  self.output:resizeAs(input)
  self.flip_idx = (self.flip_idx and self.flip_idx:nElement() == T and self.flip_idx) 
                  or torch.range(T, 1, -1):long()
  local output_5dview = self.output:view(N, T, D, H, W)
  output_5dview:indexCopy(2, self.flip_idx, input_5d)
  return self.output
end


function VidFlip:updateGradInput(input, gradOutput)
  local K, D, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  local N, T = self.N, self.T
  assert(K == N * T, 'Total number of image does not match.')
  local gradOutput_5d = gradOutput:view(N, T, D, H, W)
  self.gradInput = (self.gradInput:nElement() > 0 and self.gradInput) or gradOutput:clone():zero()
  self.gradInput:resizeAs(gradOutput)
  self.flip_idx = (self.flip_idx and self.flip_idx:nElement() == T and self.flip_idx) 
                  or torch.range(T, 1, -1):long()
  local gradInput_5dview = self.gradInput:view(N, T, D, H, W)
  gradInput_5dview:indexCopy(2, self.flip_idx, gradOutput_5d)
  return self.gradInput
end
