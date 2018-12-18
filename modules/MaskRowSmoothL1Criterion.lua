
local MaskRowSmoothL1Criterion, parent = torch.class('nn.MaskRowSmoothL1Criterion', 'nn.SmoothL1Criterion')

function MaskRowSmoothL1Criterion:updateOutput(inputs, targets)
   local target = targets[1] 
   local indicator = targets[2]
   self._buffer2 = self._buffer2 or inputs.new()
   self._buffer2:resizeAs(inputs):copy(inputs)
   local zero_idx = indicator:eq(0):nonzero()
   if zero_idx:nElement() > 0 then
    zero_idx = zero_idx:view(-1):long()
    self._buffer2:indexCopy(1, zero_idx, target:index(1, zero_idx))
   end
   parent.updateOutput(self, self._buffer2, target)
   local B = torch.sum(indicator)
   self.output = self.output / (B + 1e-10)
   return self.output
end

function MaskRowSmoothL1Criterion:updateGradInput(inputs, targets)
   local target = targets[1] 
   local indicator = targets[2]
   local B = torch.sum(indicator)
   parent.updateGradInput(self, self._buffer2, target)
   return self.gradInput:div(B + 1e-10)
end
