--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local loader = require 'loaders.loader'

local function DataLoader(dset, data_root)
   if torch.typename(dset) == 'dataLoader' then return dset end
   local file = paths.concat(data_root, 'exp', 'annotations', dset .. '.json')
   if not file then
      error('invalid dataset: ' .. tostring(dset))
   end
   local img_path
   if string.find(dset, 'train') ~= nil then
      img_path = paths.concat(data_root, 'Data', 'DET', 'train')
   elseif string.find(dset, 'val') ~= nil then
      img_path = paths.concat(data_root, 'Data', 'DET', 'val')
   end
   return loader():load(file, img_path)
end

return DataLoader
