--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local tnt = require 'torchnet'
require 'donkey_static'

-- create an instance of DataSetJSON to make roidb and scoredb
-- that are passed to threads
local roidb, scoredb
do
   local ds = loadDataSet(static_opt)
   ds:loadROIDB(static_opt.best_proposals_number)
   roidb, scoredb = ds.roidb, ds.scoredb
end
local loader = createTrainLoader(static_opt, roidb, scoredb, 1)
local bbox_regr = loader:setupData()
local epoch_size = math.ceil(loader.dataset.dataset:nImages() / static_opt.images_per_batch)
static_opt.epochSize = epoch_size
local local_opt = tnt.utils.table.clone(static_opt)

local function getParallelIterator()
  return tnt.ParallelDatasetIterator{
     nthread = local_opt.nDonkeys,
     init = function(idx)
        require 'torchnet'
        require 'donkey_static'
        torch.manualSeed(local_opt.manualSeed + idx)
        g_donkey_idx = idx
     end,
     closure = function()
        local loaders = {}
        loaders[1] = createTrainLoader(local_opt, roidb, scoredb, 1)
        for i,v in ipairs(loaders) do
           v.bbox_regr = bbox_regr
        end
        return tnt.ListDataset{
           list = torch.range(1, epoch_size):long(),
           load = function(idx)
              return {loaders[torch.random(#loaders)]:sample()}
           end,
        }
     end,
  }
end

local function getIterator()
  local dataset = tnt.ListDataset{
                    list = torch.range(1, epoch_size):long(),
                    load = function(idx)
                       return {loader:sample()}
                    end,
                  }
  local iterator = tnt.DatasetIterator(dataset)
  return iterator
end

return {getIterator, getParallelIterator, loader}


