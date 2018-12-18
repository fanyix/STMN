local PSROIPooling, parent = torch.class('rfcn.PSROIPooling', 'nn.Module')
local C = rfcn.C

function PSROIPooling:__init(spatial_scale,grid_size,output_dim)
  parent.__init(self)
  self.pooled_width = grid_size
  self.pooled_height = grid_size
  self.spatial_scale = spatial_scale
  self.output_dim = output_dim
  self.gradInput = {}
  self.indices = torch.Tensor()
end

function PSROIPooling:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function PSROIPooling:updateOutput(input)
  assert(#input == 2)
  local data = input[1]
  local rois = input[2]
  local height = data:size(3)
  local width = data:size(4)
  -- invoke cuda compute
  C.PSROIPooling_updateOutput(cutorch.getState(), self.output:cdata(), self.indices:cdata(), data:cdata(), rois:cdata(), height, width, self.pooled_height, self.pooled_width, self.output_dim, self.spatial_scale)
  return self.output
end

function PSROIPooling:updateGradInput(input,gradOutput)
  assert(#input == 2)
  local data = input[1]
  local rois = input[2]
  local height = data:size(3)
  local width = data:size(4)
  self.gradInput_boxes = self.gradInput_boxes or data.new()
  self.gradInput_rois = self.gradInput_rois or data.new()
  C.PSROIPooling_updateGradInputAtomic(cutorch.getState(), self.gradInput_boxes:cdata(), gradOutput:cdata(), data:cdata(), 
    rois:cdata(), self.indices:cdata(), height, width, self.pooled_height, self.pooled_width, 
    self.output_dim, self.spatial_scale)
  self.gradInput_rois:resizeAs(rois):zero()
  self.gradInput = {self.gradInput_boxes, self.gradInput_rois}
  return self.gradInput
end

function PSROIPooling:clearState()
   nn.utils.clear(self, 'gradInput_rois', 'gradInput_boxes', 'indices')
   return parent.clearState(self)
end