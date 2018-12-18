--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'nn'

fbcoco = {}

require 'BatchProviderVID'
require 'BatchProviderBase_StaticImg'
require 'BatchProviderROI_StaticImg'

require 'Tester_VID'
require 'ImageDetect'
require 'ColorTransformer'

require 'modules.ImageTransformer'
require 'modules.ContextRegion'
require 'modules.Foveal'
require 'modules.SelectBoxes'
require 'modules.ConvertFrom'
require 'modules.BBoxRegressionCriterion'
require 'modules.NoBackprop'
require 'modules.BBoxNorm'
require 'modules.ModeSwitch'
require 'modules.SequentialSplitBatch'

require 'modules.ModelParallelTable'

return fbcoco
