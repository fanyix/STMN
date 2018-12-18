--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'nn'
require 'engines.Optim'

local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local FBOptimEngine, SGDEngine = torch.class('tnt.FBOptimEngine', 'tnt.SGDEngine', tnt)

FBOptimEngine.__init = argcheck{
   {name="self", type="tnt.FBOptimEngine"},
   call =
      function(self)
         SGDEngine.__init(self)
      end
}

FBOptimEngine.execute_train = function (self, state)
  self.hooks("onSample", state)
  assert(state.network.train)
  --state.network:forward(sample.input)
  self.hooks("onForward", state)
  --state.criterion:forward(state.network.output, sample.target)
  self.hooks("onForwardCriterion", state)
 
  assert(state.network.train)
  --state.criterion:backward(state.network.output, sample.target)
  self.hooks("onBackwardCriterion", state)
  --state.network:backward(sample.input, state.criterion.gradInput)
  self.hooks("onBackward", state)

  state.fowardBackwardCounter =  state.fowardBackwardCounter + 1
  if state.fowardBackwardCounter >= state.forwardBackwardPerUpdate then
    state.optimizer:updateParameters(state.optimMethod, state.criterion.output)
    state.t = state.t + 1
    self.hooks("onUpdate", state)
    state.network:zeroGradParameters()
    if state.criterion.zeroGradParameters then
       state.criterion:zeroGradParameters()
    end
    state.fowardBackwardCounter = 0
  end
end

FBOptimEngine.train = argcheck{
   {name="self", type="tnt.FBOptimEngine"},
   {name="network", type="nn.Module"},
   {name="criterion", type="nn.Criterion"},
   {name="iterator", type="table"},
   {name="maxepoch", type="number", default=1000},
   {name="optimMethod", type="function"},
   {name="config", type="table", opt=true},
   call =
      function(self, network, criterion, iterator, maxepoch, optimMethod, config)
         local state = {
            network = network,
            criterion = criterion,
            iterator = iterator,
            maxepoch = maxepoch,
            optimMethod = optimMethod,
            optimizer = nn.Optim(network, config),
            config = config,
            sample = {},
            epoch = 0, -- epoch done so far
            t = 0, -- samples seen so far
            forward_backward_iter = 0,
            training = true,
            forwardBackwardPerUpdate = config.forwardBackwardPerUpdate
         }
         state.fowardBackwardCounter = 0
         state.network:zeroGradParameters()
         if state.criterion.zeroGradParameters then
            state.criterion:zeroGradParameters()
         end
         self.hooks("onStart", state)
         while state.epoch < state.maxepoch do
            state.network:training()
            self.hooks("onStartEpoch", state)
            for batch_iter = 1, config.epochSize do
              for sample in state.iterator['video']() do
                state.sample = sample
                state.n = state.n + 1
                self:execute_train(state)
                break
              end
              if config.use_DET then
                for sample in state.iterator['static']() do
                  if not state.utils then
                    state.utils = paths.dofile '../utils.lua'
                  end
                  state.sample = sample
                  if config.DET_train_mode == 'seq_expansion' then
                    -- expand image into sequences
                    local default_N, default_T = state.utils.set_NT(state.network, config.DET_images_per_batch, config.timestep_per_batch)
                    state.utils.img_to_seq(state.sample, config.timestep_per_batch)
                    self:execute_train(state)
                    state.utils.set_NT(state.network, default_N, default_T)
                  elseif config.DET_train_mode == 'single_frame' then
                    local default_N, default_T = state.utils.set_NT(state.network, sample[1][1]:size(1), 1)
                    self:execute_train(state)
                    state.utils.set_NT(state.network, default_N, default_T)
                  else
                    assert(false, 'Unknown DET_train_mode flag.')
                  end
                  break
                end
              end
            end
            state.epoch = state.epoch + 1
            self.hooks("onEndEpoch", state)
         end
         self.hooks("onEnd", state)
      end
}
