--[[----------------------------------------------------------------------------
Train detector with videos
------------------------------------------------------------------------------]]
--package.path = package.path .. ';./multipathnet/?.lua'

require 'torch'
require 'nn'
require 'engines.fboptimengine'
require 'fbcoco'
require 'image'

local config = require 'config'
local os = require 'os'
local tnt = require 'torchnet'
local json = require 'cjson'
local utils = paths.dofile 'utils.lua'
local model_utils = paths.dofile 'models/model_utils.lua'
local mu = require 'myutils'
local gnuplot = require 'gnuplot'
--local qtwidget = require 'qtwidget'

-- compute the caller directory
local this_file_dir = paths.dirname(paths.thisfile(nil))

-- cmd options
local cmd = torch.CmdLine()
cmd:option('-model', 'stmn', 'model to train (stmn / rfcn)')
cmd:option('-ckpt', 'dev', 'name for the checkpoint')
local cmd_args = cmd:parse(arg)

-- set some directories
local data_root = paths.concat(this_file_dir, '..', 'dataset/ImageNetVID/')
local static_root = paths.concat(this_file_dir, '..', 'dataset/ImageNetDET/')
local ckpt_root = paths.concat(data_root, string.format('ckpt/%s/', cmd_args.ckpt))

-- config the main opt
opt, static_opt = config.get_train_config(cmd_args.model, data_root, static_root, ckpt_root)

if opt.fg_threshold < 0 then
   opt.fg_threshold = opt.bg_threshold_max
end
if opt.manualSeed == -1 then --random
   opt.manualSeed = torch.random(10000)
end
print(opt)
model_opt = {}

require 'cutorch'
math.randomseed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
torch.manualSeed(opt.manualSeed)

---------------------------------------------------------------------------------------
-- model
---------------------------------------------------------------------------------------
assert((opt.seq_per_batch*opt.timestep_per_batch) % opt.nGPU == 0, 
    "N*T must be a multiple of nGPU")
if opt.dataset == 'ImageNetVID' then
  opt.num_classes = 30 + 1
else
  assert(false, 'Unknown dataset.')
end

local model_data = paths.dofile(paths.concat(this_file_dir, 'models/model.lua'))
local model, transformer_path, flow_transformer_path = table.unpack(model_data)
opt.transformer = transformer_path
static_opt.transformer = transformer_path

if opt.train_remove_dropouts then
   model_utils.removeDropouts(model)
end

-- load transformer 
local transformer = torch.load(opt.transformer)

-- create a video image loader
local getIterator, getParallelIterator, createVideoTrainLoader, loader = table.unpack(require 'data_video')
local iterator = getParallelIterator()  -- getIterator(), getParallelIterator()

-- create a static image loader
local getIterator_static, getParallelIterator_static, loader_static = table.unpack(require 'data_static')
local iterator_static = getParallelIterator_static()

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
  
  model_utils.loadModelWeights(loaded_model, model, opt.load_mode)
end

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

-- make a loader for validation data
local anno = torch.load(opt.val_anno_file)
local loader_opt = tnt.utils.table.clone(opt)
loader_opt.prop_dir = loader_opt.val_prop_dir
loader_opt.img_dir = loader_opt.val_img_dir
if opt.eval_timestep_per_batch then
  loader_opt.timestep_per_batch = opt.eval_timestep_per_batch
end
opt.eval_seq_per_batch = 1
if opt.eval_seq_per_batch then
  loader_opt.seq_per_batch = opt.eval_seq_per_batch
end
loader_opt.aspect_ratio_sorting = false
-- cancel out all data augmentation
loader_opt.brightness_var = nil
loader_opt.contrast_var = nil 
loader_opt.saturation_var = nil
loader_opt.lighting_var = nil
loader_opt.scale_jitter = 0.0  -- default to 0
loader_opt.aspect_jitter = 0.0  -- default to 0
local val_loader = createVideoTrainLoader(loader_opt, anno)
val_loader.bbox_regr = g_mean_std
local tester = fbcoco.Tester_VID(model, transformer, val_loader, {loader_opt.scale}, loader_opt.max_size, loader_opt)


-- also assign box reg mean/std to train loader
loader.bbox_regr.mean:copy(g_mean_std.mean)
loader.bbox_regr.std:copy(g_mean_std.std)

--------------------------------------------------------------------------
-- training
--------------------------------------------------------------------------

local samples = {}
local train_loss = mu.init_recorder(100)
local val_loss = mu.init_recorder(10)


local function createCriterion()
   local criterion = nn.ParallelCriterion()
   :add(nn.CrossEntropyCriterion(), 1)
   :add(nn.BBoxRegressionCriterion(), opt.bbox_regression)
   return criterion:cuda()
end

local dataTimer = tnt.TimeMeter()
local timer, batchTimer = tnt.TimeMeter({ unit = true }), tnt.TimeMeter()

local engine = tnt.FBOptimEngine()
local function json_log(t) print('json_stats: '..json.encode(t)) end

-----------------------------------------------------------------------------

local function save(model, state, g_mean_std, epoch)
   opt.test_model = 'model_'..epoch..'.t7'
   opt.test_state = 'optimState_'..epoch..'.t7'
   local model_path = paths.concat(opt.save_folder, opt.test_model)
   local state_path = paths.concat(opt.save_folder, opt.test_state)
   paths.mkdir(paths.dirname(model_path))
   print("Saving model to "..model_path)
   if opt.checkpoint_mode == 'weights' then
    torch.save(model_path, {model=utils.weights_checkpoint(model), g_mean_std=g_mean_std})
   elseif opt.checkpoint_mode == 'binary' then
    torch.save(model_path, {model=utils.checkpoint(model), g_mean_std=g_mean_std})
   elseif opt.checkpoint_mode == 'weights_plus_STMM' then
    local STMM = model:findModules('nn.STMM')
    torch.save(model_path, {model=utils.weights_checkpoint(model), STMM=STMM, g_mean_std=g_mean_std})
   else
    assert(false, 'Unknown checkpoint mode.')
   end
   collectgarbage()
   --print("Saving state to "..state_path)
   --torch.save(state_path, state)
end

local function validate(state)
   local compact_eval_mode = false
   local val_loader_batch_N
   local model = state.network
   local iter = state.epoch
   local res = nil
   local eval_toolkit_dir = paths.concat(this_file_dir, 'external/VID_devkit/evaluation')
   local eval_toolkit_data_dir = paths.concat(this_file_dir, 'external/VID_devkit_data')
   model:evaluate()
   model:clearState()
   if compact_eval_mode then
     -- this ensures that during testing it does not use too much mem
     val_loader_batch_N = tester.loader.batch_N
     tester.loader.batch_N = 1 
   end
   -- test
   tester:test(iter, opt.eval_res_dir, data_root, eval_toolkit_dir, eval_toolkit_data_dir)
   if compact_eval_mode then
     -- reset the batch_N for val_loader
     tester.loader.batch_N = val_loader_batch_N
   end
   model:clearState()
   model:training()
   return res
end

engine.hooks.onStart = function(state)
   state.learningRate = opt.learningRate
   state.decay = opt.decay
   state.step = opt.step
   utils.cleanupOptim(state)
   if opt.checkpoint then
      local filename = checkpoint.resume(state)
      if filename then
         print("WARNING: restarted from checkpoint:", filename)
      elseif opt.resume ~= '' then
         print("resuming from checkpoint:", opt.resume)
         checkpoint.apply(state, opt.resume)
      end
   end
end

engine.hooks.onStartEpoch = function(state)
   local epoch = state.epoch + 1
   if opt.checkpoint and epoch % opt.snapshot == 0 then
      checkpoint.checkpoint(state, opt)
   end
   print("Training epoch " .. epoch .. "/" .. opt.nEpochs)
   timer:reset()
   state.n = 0
end

engine.hooks.onSample = function(state)
   cutorch.synchronize(); 
   collectgarbage();
   dataTimer:stop()
   utils.recursiveCast(samples, state.sample, 'torch.CudaTensor')
   state.sample.input = samples[1]
   state.sample.target = samples[2]
end

local post_stack_output = {torch.CudaTensor(), torch.CudaTensor()}
local conv_stack_output = torch.CudaTensor()
local STMM_stack_output = torch.CudaTensor()
engine.hooks.onForward = function(state)
  if not opt.memory_optimization then
    if opt.backprop_batch_size and opt.backprop_batch_size ~= opt.batch_size then
      -- forward all proposals as candidate
      state.network:evaluate()
      local network = state.network
      local sample = state.sample
      local frm_idx = sample.input[2]:select(2, 1)
      local T = sample.input[1]:size(1)
      --local B = T / opt.nGPU
      local proc_bs = T
      local proc_B = T / proc_bs
      local final_idx = {}
      local start_idx, end_idx = 0, 0 
      for idx = 1, proc_B do
        start_idx = end_idx + 1
        end_idx = start_idx + proc_bs - 1
        local feat = sample.input[1][{{start_idx, end_idx}, {}, {}, {}}]
        local box_idx = torch.cmul(frm_idx:ge(start_idx), frm_idx:le(end_idx)):nonzero():view(-1)
        local box = sample.input[2]:index(1, box_idx)
        box:select(2, 1):csub(start_idx - 1)
        local gt_class = sample.target[1]:index(1, box_idx)
        local out = network:forward({feat, box})
        for scan_idx = 1, end_idx-start_idx+1 do
          local scan_indicator = box:select(2, 1):eq(scan_idx):nonzero():view(-1)
          local scan_box = box:index(1, scan_indicator):narrow(2, 2, 4)
          local scan_score = out[1]:index(1, scan_indicator)
          local scan_class = gt_class:index(1, scan_indicator)
          local sampled_idx
          if opt.mine_hard_algo == 'ohem' then
            sampled_idx = loader:mine_hard_neg_v2(scan_box, scan_score, scan_class, opt)
          elseif opt.mine_hard_algo == 'balanced' then
            sampled_idx = loader:mine_hard_neg(scan_score, scan_class, opt)
          else
            assert(false, 'Unknown opt.mine_hard_algo')
          end
          sampled_idx = box_idx:index(1, scan_indicator:index(1, sampled_idx))
          table.insert(final_idx, sampled_idx)
        end
      end     
      final_idx = torch.cat(final_idx, 1)
      state.network:training()
      
      -- trim samples
      sample.input[2] = sample.input[2]:index(1, final_idx)
      sample.target[1] = sample.target[1]:index(1, final_idx)
      sample.target[2][1] = sample.target[2][1]:index(1, final_idx)
      sample.target[2][2] = sample.target[2][2]:index(1, final_idx)
      
      -- perform forward again with hard negatives
      network:forward(sample.input)
    else
      state.network:forward(state.sample.input)
    end
  else
    local network = state.network
    local conv_stack = state.network.conv_stack
    local post_stack = state.network.post_stack
    local STMM_stack = state.network.STMM_stack
    local sample = state.sample
    local T = sample.input[1]:size(1)
    local B = T / opt.nGPU
    local frm_idx = sample.input[2]:select(2, 1)
    local box_n = sample.input[2]:size(1)
    local start_idx, end_idx = 0, 0 
    for idx = 1, B do
      start_idx = end_idx + 1
      end_idx = start_idx + opt.nGPU - 1
      local feat = sample.input[1][{{start_idx, end_idx}, {}, {}, {}}]
      local output = conv_stack:forward(feat)
      if idx == 1 then
        conv_stack_output:resize(T, output:size(2), output:size(3), output:size(4)):zero()
      end
      conv_stack_output:narrow(1, start_idx, end_idx - start_idx + 1):copy(output)
    end
    
    local output = STMM_stack:forward(conv_stack_output)
    STMM_stack_output:resizeAs(output):copy(output)
    
    if opt.backprop_batch_size and opt.backprop_batch_size ~= opt.batch_size then
      -- forward all proposals as candidate
      state.network:evaluate()
      local start_idx, end_idx = 0, 0 
      local final_idx = {}
      local proc_bs = 1
      local proc_B = T / proc_bs
      for idx = 1, proc_B do
        start_idx = end_idx + 1
        end_idx = start_idx + proc_bs - 1
        local feat = STMM_stack_output[{{start_idx, end_idx}, {}, {}, {}}]
        local box_idx = torch.cmul(frm_idx:ge(start_idx), frm_idx:le(end_idx)):nonzero():view(-1)
        local box = sample.input[2]:index(1, box_idx)
        box:select(2, 1):csub(start_idx - 1)
        local gt_class = sample.target[1]:index(1, box_idx)
        local out = post_stack:forward({feat, box})
        for scan_idx = 1, end_idx-start_idx+1 do
          local scan_indicator = box:select(2, 1):eq(scan_idx):nonzero():view(-1)
          local scan_box = box:index(1, scan_indicator):narrow(2, 2, 4)
          local scan_score = out[1]:index(1, scan_indicator)
          local scan_class = gt_class:index(1, scan_indicator)
          local sampled_idx = loader:mine_hard_neg(scan_score, scan_class, opt)
          sampled_idx = box_idx:index(1, scan_indicator:index(1, sampled_idx))
          table.insert(final_idx, sampled_idx)
        end
      end
      final_idx = torch.cat(final_idx, 1)
      state.network:training()
      -- trim samples
      sample.input[2] = sample.input[2]:index(1, final_idx)
      sample.target[1] = sample.target[1]:index(1, final_idx)
      sample.target[2][1] = sample.target[2][1]:index(1, final_idx)
      sample.target[2][2] = sample.target[2][2]:index(1, final_idx)
      -- perform forward again with hard negatives
      frm_idx = sample.input[2]:select(2, 1)
      box_n = sample.input[2]:size(1)
      post_stack_output[1]:resize(box_n, opt.num_classes):zero()
      post_stack_output[2]:resize(box_n, opt.num_classes*4):zero()
      local start_idx, end_idx = 0, 0 
      for idx = 1, B do
        start_idx = end_idx + 1
        end_idx = start_idx + opt.nGPU - 1
        local feat = STMM_stack_output[{{start_idx, end_idx}, {}, {}, {}}]
        local box_idx = torch.cmul(frm_idx:ge(start_idx), frm_idx:le(end_idx)):nonzero():view(-1)
        local box = sample.input[2]:index(1, box_idx)
        box:select(2, 1):csub(start_idx - 1)
        local out = post_stack:forward({feat, box})
        post_stack_output[1]:indexCopy(1, box_idx, out[1])
        post_stack_output[2]:indexCopy(1, box_idx, out[2])
      end
      network.output = post_stack_output
      
    else
      post_stack_output[1]:resize(box_n, opt.num_classes):zero()
      post_stack_output[2]:resize(box_n, opt.num_classes*4):zero()
      local start_idx, end_idx = 0, 0 
      for idx = 1, B do
        start_idx = end_idx + 1
        end_idx = start_idx + opt.nGPU - 1
        local feat = STMM_stack_output[{{start_idx, end_idx}, {}, {}, {}}]
        local box_idx = torch.cmul(frm_idx:ge(start_idx), frm_idx:le(end_idx)):nonzero():view(-1)
        local box = sample.input[2]:index(1, box_idx)
        box:select(2, 1):csub(start_idx - 1)
        local out = post_stack:forward({feat, box})
        post_stack_output[1]:indexCopy(1, box_idx, out[1])
        post_stack_output[2]:indexCopy(1, box_idx, out[2])
      end
      network.output = post_stack_output
    end
  end
  
  -- clear memory
  collectgarbage()
end

local post_stack_gradInput = torch.CudaTensor()
engine.hooks.onBackward = function(state)  
  if not opt.memory_optimization then
    state.network:backward(state.sample.input, state.criterion.gradInput)
  else
    local conv_stack = state.network.conv_stack
    local post_stack = state.network.post_stack
    local STMM_stack = state.network.STMM_stack
    local sample = state.sample
    local T = sample.input[1]:size(1)
    local B = T / opt.nGPU
    local frm_idx = sample.input[2]:select(2, 1)
    
    local start_idx, end_idx = 0, 0 
    post_stack_gradInput:resizeAs(STMM_stack_output):zero()
    for idx = 1, B do
      start_idx = end_idx + 1
      end_idx = start_idx + opt.nGPU - 1
      local feat = STMM_stack_output[{{start_idx, end_idx}, {}, {}, {}}]
      local box_idx = torch.cmul(frm_idx:ge(start_idx), frm_idx:le(end_idx)):nonzero():view(-1)
      local box = sample.input[2]:index(1, box_idx)
      box:select(2, 1):csub(start_idx - 1)
      local gradOutput_1 = state.criterion.gradInput[1]:index(1, box_idx)
      local gradOutput_2 = state.criterion.gradInput[2]:index(1, box_idx)
      local gradInput = post_stack_gradInput:narrow(1, start_idx, end_idx - start_idx + 1)
      post_stack:forward({feat, box})
      gradInput:copy(post_stack:backward({feat, box}, {gradOutput_1, gradOutput_2})[1])
    end
    
    local STMM_stack_gradInput = STMM_stack:backward(conv_stack_output, post_stack_gradInput)
  
    local start_idx, end_idx = 0, 0 
    for idx = 1, B do
      start_idx = end_idx + 1
      end_idx = start_idx + opt.nGPU - 1
      local feat = sample.input[1][{{start_idx, end_idx}, {}, {}, {}}]
      local gradOutput = STMM_stack_gradInput[{{start_idx, end_idx}, {}, {}, {}}]
      conv_stack:forward(feat)
      conv_stack:backward(feat, gradOutput)
    end
  end
  
  -- clear memory
  collectgarbage()
end

engine.hooks.onForwardCriterion = function(state)
  state.criterion:forward(state.network.output, state.sample.target)
end

engine.hooks.onBackwardCriterion = function(state)
  state.criterion:backward(state.network.output, state.sample.target)
end


engine.hooks.onUpdate = function(state)
   cutorch.synchronize(); 
   collectgarbage();

   local err = state.criterion.output   
   train_loss = mu.update_loss(train_loss, err, state.forward_backward_iter)

   timer:incUnit()

   print(('Epoch: [%d][%d/%d]\tTime %.3f (%.3f) DataTime %.3f Err %.4f'):format(
   state.epoch + 1, state.n, opt.epochSize, batchTimer:value(), timer:value(), dataTimer:value(), err))

   dataTimer:reset()
   dataTimer:resume()
   batchTimer:reset()
   state.forward_backward_iter = state.forward_backward_iter + 1
   
   if state.forward_backward_iter % 500 == 0 then
    os.execute('nvidia-smi')
    state.network:clearState()
    state.network:training()
   end
   
   
   if opt.plot_loss_every > 0 and state.forward_backward_iter % opt.plot_loss_every == 0 then
    local tr_loss, tr_iter = mu.retrieve_loss(train_loss)
    local va_loss, va_iter = mu.retrieve_loss(val_loss)
    local loss_file = opt.loss_plot_file
    paths.mkdir(paths.dirname(loss_file))
    gnuplot.svgfigure(loss_file)
    if #tr_loss > 0 and #va_loss > 0 then
      gnuplot.plot({'train', torch.Tensor(tr_iter), torch.Tensor(tr_loss), '-'},
                    {'val', torch.Tensor(va_iter), torch.Tensor(va_loss), '-'})
    elseif #tr_loss > 0 then
      gnuplot.plot({'train', torch.Tensor(tr_iter), torch.Tensor(tr_loss), '-'})
    end
    gnuplot.xlabel('Iteration')
    gnuplot.ylabel('Loss')
    gnuplot.plotflush()
    gnuplot.closeall()
   end
  
  -- let's do a gradient profile
  if opt.gradient_profile_every > 0 and state.forward_backward_iter % opt.gradient_profile_every == 0 then
    local net_to_be_profiled = {
                net = state.network
    }
    local grad_profile_file = string.format('%s/%d.txt', opt.gradient_profile_dir, state.forward_backward_iter)
    paths.mkdir(paths.dirname(grad_profile_file))
    local fd = io.open(grad_profile_file, 'w')
    for net_name, v in pairs(net_to_be_profiled) do
      local db_params, db_grad_params = v:parameters()
      for db_idx = 1, #db_params do
        local param_mag = torch.norm(db_params[db_idx])
        local grad_mag = torch.norm(db_grad_params[db_idx])
        local ratio = grad_mag / param_mag
        local grad_str = string.format('%s, blob %d ([%s]), |param|=%g, |grad|=%g, ratio=%g\n\n',
              net_name, db_idx, mu.size_string(db_params[db_idx]),
              param_mag, grad_mag, ratio)
        fd:write(grad_str)
      end
    end
    fd:close()
  end
  
  -- compute the validation loss
  if opt.val_loss_every > 0 and state.forward_backward_iter % opt.val_loss_every == 0 then
    state.network:evaluate()
    local default_N, default_T = utils.set_NT(state.network, val_loader.batch_N, val_loader.batch_T)  
    local loss = 0
    for val_forward_iter = 1, opt.val_forward_maxiter do
      local sample = {val_loader:sample()}
      state.sample = sample
      engine.hooks("onSample", state)
      engine.hooks("onForward", state)
      engine.hooks("onForwardCriterion", state)
      loss = loss + state.criterion.output      
    end
    loss = loss / opt.val_forward_maxiter
    val_loss = mu.update_loss(val_loss, loss, state.forward_backward_iter)
    utils.set_NT(state.network, default_N, default_T)
    state.network:training()
  end
end

engine.hooks.onEndEpoch = function(state)
   local epoch = state.epoch + 1
   if epoch % state.step == 0 then
      print('Dropping learning rate')
      state.learningRate = state.learningRate * state.decay
      local optimizer = state.optimizer
      for k,v in pairs(optimizer.modulesToOptState) do if v[1] then
         for i,u in ipairs(v) do
            if u.dfdx then
               local curdev = cutorch.getDevice()
               cutorch.setDevice(u.dfdx:getDevice())
               u.dfdx:mul(state.decay)
               cutorch.setDevice(curdev)
               u.learningRate = u.learningRate * state.decay
            end
         end
      end end
   end
   if epoch % opt.snapshot == 0 then   
      save(state.network, state.optimizer, g_mean_std, epoch)
      local res = validate(state)
      -- GPU memory profiling
      os.execute('nvidia-smi')
   end
end

engine.hooks.onEnd = function(state)
   print("Done training. Running final validation")
   save(state.network, state.optimizer, g_mean_std, 'final')
   local res = validate(state)
   -- GPU memory profiling
   os.execute('nvidia-smi')
end

local iterators = {video = iterator, static = iterator_static}
engine:train{
   network = model,
   criterion = createCriterion(),
   config = opt,
   maxepoch = opt.nEpochs,
   optimMethod = opt.method,
   iterator = iterators,
}






