require 'torch'
local C = assemble.C

local function gpu_assemble(cur_prev_aff,feat,output,masked_cpa,pad)
  C.assemble_engine(cutorch.getState(),
                    cur_prev_aff:cdata(),
                    feat:cdata(),
                    output:cdata(),
                    masked_cpa:cdata(),
                    pad)
end
rawset(assemble, 'gpu_assemble', gpu_assemble)
