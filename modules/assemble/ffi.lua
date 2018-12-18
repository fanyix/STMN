local ffi = require 'ffi'

local libpath = package.searchpath('libassemble', package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
void assemble_engine(THCState* state,
		            THCudaTensor* cur_prev_aff,
		            THCudaTensor* feat,
		            THCudaTensor* output,
		            THCudaTensor* masked_cpa,
		            int pad);
]]

return ffi.load(libpath)
