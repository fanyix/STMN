-- This is a useful module which, clean out gradient when doing backpropagation

local ClearGradient, parent = torch.class('nn.ClearGradient', 'nn.Module')

function ClearGradient:__init(tag)
   parent.__init(self)
   self.cut_tag = tag
   self.gradInput = nil
end

function ClearGradient:updateOutput(input)
   self.output = input
   return self.output
end

function ClearGradient:updateGradInput(input, gradOutput)
 if torch.isTensor(input) then
    self.gradInput = self.gradInput or input.new()
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
  else
    for idx = 1, #input do
      self.gradInput[idx] = self.gradInput[idx] or input[idx].new()
      self.gradInput[idx]:resizeAs(input[idx])
      self.gradInput[idx]:zero()
    end
  end
  return self.gradInput
end