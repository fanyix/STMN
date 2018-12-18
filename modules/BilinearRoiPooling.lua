require 'torch'
require 'nn'

require 'stn'
require 'modules.BatchBilinearSamplerBHWD'
require 'modules.BoxToAffine'

local layer, parent = torch.class('nn.BilinearRoiPooling', 'nn.Module')

--[[
BilinearRoiPooling is a layer that uses bilinear sampling to pool featurs for a
region of interest (RoI) into a fixed size.

The constructor takes inputs height and width, both integers giving the size to
which RoI features should be pooled. For example if RoI feature maps are being
fed to VGG-16 fully connected layers, then we should have height = width = 7.

WARNING: The bounding box coordinates given in the forward pass should be in
the coordinate system of the input image used to compute the feature map, NOT
in the coordinate system of the feature map. To properly compute the forward
pass, the module needs to know the size of the input image; therefore the method
setImageSize(image_height, image_width) must be called before each forward pass.

Inputs:
- feats: Tensor of shape (C, H, W) giving a convolutional feature map.
- boxes: Tensor of shape (B, 4) giving bounding box coordinates in
         (xc, yc, w, h) format; the bounding box coordinates are in
         coordinate system of the original image, NOT the convolutional
         feature map.

Return:
- roi_features:
--]]

function layer:__init(height, width, spatial_scale)
  parent.__init(self)
  self.height = height
  self.width = width
  self.spatial_scale = spatial_scale
  self.gradInput = {}
  
  -- box_branch is a net to convert box coordinates of shape (B, 4)
  -- to sampling grids of shape (B, height, width)
  self.box_branch = nn.Sequential()

  -- box_to_affine converts boxes of shape (B, 4) to affine parameter
  -- matrices of shape (B, 2, 3); on each forward pass we need to call
  -- box_to_affine:setSize() to set the size of the input image.
  self.box_to_affine = nn.BoxToAffine()
  self.box_branch:add(self.box_to_affine)

  -- Grid generator converts matrices to sampling grids of shape
  -- (B, height, width, 2).
  self.box_branch:add(nn.AffineGridGeneratorBHWD(self.height * 2, self.width * 2))

  self.net = nn.Sequential()
  local parallel = nn.ParallelTable()
  parallel:add(nn.Transpose({1, 2}, {2, 3}))
  parallel:add(self.box_branch)
  self.net:add(parallel)  
  self.net:add(nn.BatchBilinearSamplerBHWD())
  self.net:add(nn.Transpose({3, 4}, {2, 3}))
  self.net:add(cudnn.SpatialAveragePooling(2, 2, 2, 2, 0, 0))
end


function layer:clearState()
  self.net:clearState()
end


function layer:updateOutput(input)
  local feats = input[1]
  local boxes = input[2][{{}, {2, 5}}]:clone()
  boxes[{{}, 3}]:csub(boxes[{{}, 1}]):add(1)
  boxes[{{}, 4}]:csub(boxes[{{}, 2}]):add(1)
  boxes[{{}, 1}]:add(boxes[{{}, 3}]/2)
  boxes[{{}, 2}]:add(boxes[{{}, 4}]/2)
  local box_img_idx = input[2][{{}, 1}]
  local image_num = feats:size(1)
  local image_dim = feats:size(2)
  local image_height = feats:size(3) / self.spatial_scale
  local image_width = feats:size(4) / self.spatial_scale
  local box_num = boxes:size(1)
  self.box_to_affine:setSize(image_height, image_width)
  self.output = self.output or feats.new()
  self.output:resize(box_num, image_dim, self.height, self.width):zero()
  
  for img_idx = 1, image_num do
  	local val_box_idx = box_img_idx:eq(img_idx):nonzero()
  	if val_box_idx:nElement() > 0 then
  		val_box_idx = val_box_idx:view(-1)
	  	local cur_feats = feats[{img_idx, {}, {}, {}}]
	  	local cur_boxes = boxes:index(1, val_box_idx)
	  	local output = self.net:forward({cur_feats, cur_boxes})
	  	self.output:indexCopy(1, val_box_idx, output)
  	end
  end
  
  return self.output
end


function layer:updateGradInput(input, gradOutput)
  local feats = input[1]
  local boxes = input[2][{{}, {2, 5}}]:clone()
  boxes[{{}, 3}]:csub(boxes[{{}, 1}]):add(1)
  boxes[{{}, 4}]:csub(boxes[{{}, 2}]):add(1)
  boxes[{{}, 1}]:add(boxes[{{}, 3}]/2)
  boxes[{{}, 2}]:add(boxes[{{}, 4}]/2)
  local box_img_idx = input[2][{{}, 1}]
  local image_num = feats:size(1)
  local image_dim = feats:size(2)
  local image_height = feats:size(3) / self.spatial_scale
  local image_width = feats:size(4) / self.spatial_scale
  local box_num = boxes:size(1)
  self.box_to_affine:setSize(image_height, image_width)
  self.gradFeats = self.gradFeats or feats.new()
  self.gradFeats:resizeAs(feats):zero()
  self.gradRois = self.gradRois or input[2].new()
  self.gradRois:resizeAs(input[2]):zero()
  self.gradInput = self.gradInput or {} 

  for img_idx = 1, image_num do
  	local val_box_idx = box_img_idx:eq(img_idx):nonzero()
  	if val_box_idx:nElement() > 0 then
  		val_box_idx = val_box_idx:view(-1)
	  	local cur_feats = feats[{img_idx, {}, {}, {}}]
	  	local cur_boxes = boxes:index(1, val_box_idx)
	  	local cur_gradOutput = gradOutput:index(1, val_box_idx)
	  	-- we need to perform this forward to convert boxes into grids
	  	--self.box_branch:forward(cur_boxes)
	  	self.net:forward({cur_feats, cur_boxes})
	  	local gradInput = self.net:backward({cur_feats, cur_boxes}, cur_gradOutput)
	  	self.gradFeats[{img_idx, {}, {}, {}}]:copy(gradInput[1])
  	end
  end

  self.gradInput[1] = self.gradFeats
  self.gradInput[2] = self.gradRois
  return self.gradInput
end
