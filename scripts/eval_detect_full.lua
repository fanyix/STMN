--[[----------------------------------------------------------------------------
Evaluate a trained detector
------------------------------------------------------------------------------]]

package.path = package.path .. ';../?.lua'

require 'torch'
require 'nn'
require 'optim'
require 'xlua'
require 'engines.fboptimengine'
require 'fbcoco'

local tnt = require 'torchnet'
local json = require 'cjson'
local utils = paths.dofile '../utils.lua'
local model_utils = paths.dofile '../models/model_utils.lua'
local mu = require 'myutils'
local gnuplot = require 'gnuplot'
local config = require 'config'
--local qtwidget = require 'qtwidget'

-- cmd options
local cmd = torch.CmdLine()
cmd:option('-task', 1, 'Task number used in parallelization')
cmd:option('-model', 'stmn', 'model to train (stmn / rfcn)')
cmd:option('-model_path', '', 'path of the model for evaluation')
cmd:option('-ckpt', 'dev', 'name for the checkpoint')
local cmd_args = cmd:parse(arg)
local this_file_dir = paths.dirname(paths.thisfile(nil))

-- set directories
local data_root = paths.concat(this_file_dir, '../../dataset/ImageNetVID/')
local ckpt_root = paths.concat(data_root, string.format('ckpt/%s/', cmd_args.ckpt))

opt = config.get_eval_config(cmd_args.model, cmd_args.model_path, data_root, ckpt_root)

if opt.manualSeed == -1 then --random
   opt.manualSeed = torch.random(10000)
end
print(opt)
model_opt = {}

require 'cutorch'

---------------------------------------------------------------------------------------
-- model
---------------------------------------------------------------------------------------
if opt.dataset == 'ImageNetVID' then 
  opt.num_classes = 31
else
  assert(false, 'Unknown dataset.')
end

-- compute the caller directory
local model_data = paths.dofile(paths.concat(this_file_dir, '../models/model.lua'))
local model, transformer_path, info = table.unpack(model_data)
opt.transformer = transformer_path

-- remove all dropouts
model_utils.removeDropouts(model)

-- load transformer
local transformer = torch.load(opt.transformer)

-- This is borrowed from data_video
local function createVideoLoader(opt, anno)
  local transformer = torch.load(opt.transformer)
  local bp_opt = tnt.utils.table.clone(opt)
  local bp = fbcoco.BatchProviderVID(anno, transformer, bp_opt)
  bp.class_specific = opt.train_class_specific
  return bp
end


if opt.retrain ~= 'no' then
  print('Loading a retrain model:'..opt.retrain)  
  local archive = torch.load(opt.retrain)
   local loaded_model
   if archive.model ~= nil then
     loaded_model = archive.model
     g_mean_std = archive.g_mean_std
   else
    loaded_model = archive
   end
   model_utils.loadModelWeights(loaded_model, model, 'full')
end
collectgarbage()
model:cuda()

if not opt.bbox_mask_1d then
  if model.post_roi then
    model_utils.addBBoxNorm(model.post_roi, g_mean_std)
  elseif model.post_stack then
    model_utils.addBBoxNorm(model.post_stack, g_mean_std)
  else
    model_utils.addBBoxNorm(model, g_mean_std)
  end
end

local dpt = model:findModules('nn.DataParallelTable')
for idx, mod in ipairs(dpt) do
  mod:syncParameters()
end

model_utils.testModel(model)

-- seeding the rand generator
math.randomseed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- make a loader for test data
local anno = torch.load(opt.anno_file)
local loader_opt = tnt.utils.table.clone(opt)
if opt.eval_timestep_per_batch then
  loader_opt.timestep_per_batch = opt.eval_timestep_per_batch
end
local test_loader = createVideoLoader(loader_opt, anno)
test_loader.bbox_regr = g_mean_std
local tester = fbcoco.Tester_VID(model, transformer, test_loader, {opt.scale}, opt.max_size, opt)

-- detection visualization
model:evaluate()
local tag = string.format('test_%d', cmd_args.task)
tester:test_FULL(tag, opt.eval_res_dir)





