local FixedIndex, parent = torch.class('nn.FixedIndex', 'nn.Module')

function FixedIndex:__init(dimension, index)
    parent.__init(self)
    self.dimension = dimension
    self.index = index
end

function FixedIndex:updateOutput(input)
    self.output:index(input, self.dimension, self.index)
    return self.output
end

function FixedIndex:updateGradInput(input, gradOutput)
    local gradInput = self.gradInput -- no gradient for the index variable
    gradInput:resizeAs(input):zero()
    gradInput:indexAdd(self.dimension, self.index, gradOutput)
    return self.gradInput
end

function FixedIndex:clearState()
    self.gradInput:set()
    self.output:set()
    return self
end