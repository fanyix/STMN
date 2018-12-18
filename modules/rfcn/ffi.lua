local ffi = require 'ffi'

local libpath = package.searchpath('librfcn', package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
void PSROIPooling_updateOutput(THCState *state, THCudaTensor *output, THCudaTensor *indices, THCudaTensor *data, THCudaTensor* rois, int spatial_scale, int height, int width, int pooled_height, int pooled_width, int output_dim);
void PSROIPooling_updateGradInputAtomic(THCState *state, THCudaTensor *gradInput, THCudaTensor *gradOutput, THCudaTensor *data, THCudaTensor* rois, THCudaTensor *indices, int spatial_scale, int height, int width, int pooled_height, int pooled_width, int output_dim);
]]

return ffi.load(libpath)
