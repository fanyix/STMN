local MaskCrossEntropyCriterion, parent = torch.class('nn.MaskCrossEntropyCriterion', 'nn.Module')

function MaskCrossEntropyCriterion:__init(weights)
  parent.__init(self)
  self.crc = nn.CrossEntropyCriterion(weights)
end

function MaskCrossEntropyCriterion:updateOutput(input, target)
  self.output = 0
  local valid_idx = torch.nonzero(target)
  if valid_idx:nElement() > 0 then
    valid_idx = valid_idx:view(-1):long()
    local val_input = input:index(1, valid_idx):contiguous()
    local val_target = target:index(1, valid_idx):contiguous()
    self.crc:updateOutput(val_input, val_target)
    self.output = self.crc.output
  end
  return self.output
end

function MaskCrossEntropyCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input):zero()
  local valid_idx = torch.nonzero(target)
  if valid_idx:nElement() > 0 then
    valid_idx = valid_idx:view(-1):long()
    local val_input = input:index(1, valid_idx):contiguous()
    local val_target = target:index(1, valid_idx):contiguous()
    self.crc:updateGradInput(val_input, val_target)
    self.gradInput:indexCopy(1, valid_idx, self.crc.gradInput)
  end
  return self.gradInput
end

