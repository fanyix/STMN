
require 'optim'
require 'xlua'

local config = {}

function config.get_train_config(model, data_root, static_root, ckpt_root)
  local vis_root = paths.concat(ckpt_root, 'stats')
  local model_root = paths.concat(ckpt_root, 'models')

  -- general opts
  local opt = {
    epoch = 1,
    proposal_dir = 'data/proposals/',
    sampling_mode = 'NONE_REDUNDANT',
    aspect_ratio_sorting = true,
    loss_plot_file = paths.concat(vis_root, 'loss.svg'),
    reg_loss_plot_file = paths.concat(vis_root, 'reg_loss_%s.svg'),
    eval_res_dir = paths.concat(vis_root, 'quant/'),
    gradient_profile_dir = paths.concat(vis_root, 'grad/'),
    vis_det_dir = paths.concat(vis_root, 'det/'),
    save_folder = model_root,
    spec_im_size = nil,
    
    -- whether to split the rois into different batches (for parallel training)
    parallel_roi_batch = 1,
    dampening = 0,
    learningRateDecay = 0,
    nEpochs = 1e4,
    epochSize = 100,
    nDonkeys = 4,
    manualSeed = 800,
    step = 300,
    decay = 0.1,
    train_remove_dropouts = false,
    checkpoint=false,
    resume='',      
    
    -- main section
    dataset = 'ImageNetVID',
    seq_per_batch = 2, -- 2
    timestep_per_batch = 6, -- 6 
    forwardBackwardPerUpdate = 1, -- 1
    batch_size = 300,  -- 300 for hard example mining, otherwise set to 128
    fg_fraction = 0.25,  -- default to 0.25
    backprop_batch_size = 128,  -- the number of rois we actually perform backprop
    mine_hard_algo = 'balanced',  -- 'ohem' or 'balanced'
    ohem_nms_thresh = 0.7,  -- default to 0.7
    sample_n_per_box = 0,
    sample_sigma = 0.05,  -- default to 0.05
    test_batch_size = 300,  -- nil refers to using all proposals
    frame_stride = 10,  -- default to 10
    test_epoch = 5,
    dropout = 0.5,
    checkpoint_mode = 'weights',  -- 'weights', 'binary', 'weights_plus_STMM'
    scale = 360,  -- 360
    max_size = 600,  -- 600
    mem_dim = 512,
    brightness_var = nil,  -- data augmentation, default 0.4 
    contrast_var = nil,  -- default 0.4 
    saturation_var = nil,  -- default 0.4  
    lighting_var = nil,  -- default 0.4 
    scale_jitter = nil,  -- default to 0.1
    aspect_jitter = nil,  -- default to 0.0
    bbox_regression = 1,
    method = optim['sgd'],  -- optim['sgd'], optim['adam'], mu.wd_sgd
    learningRate = 1e-3,
    momentum = 0.9,
    weightDecay = 0.0,  -- 0 for ResNet, 5e-4 for VGG
    snapshot = 5,  -- default to 5 
    gradient_profile_every = 100,  -- default to 100
    plot_loss_every = 50,  -- default to 50
    val_loss_every = 100,  -- default to 100
    val_forward_maxiter = 3,
    nGPU = 2, 
    memory_optimization = false,
    fg_threshold = 0.5,     -- if -1, then set to bg_threshold_max
    bg_threshold_min = 0.0,  -- Fast-RCNN use 0.1, whereas Faster-RCNN use 0
    bg_threshold_max = 0.5,
    use_DET = true,
    DET_train_mode = 'single_frame', -- seq_expansion, single_frame
    DET_images_per_batch = 4,
    imagenet_det_subset = 'train',  -- 'train', 'subtrain'
    anno_file = paths.concat(data_root, 'exp/anno/train.t7'),  -- train_debug.t7, train.t7
    img_dir = paths.concat(data_root, 'Data/VID/train'),
    prop_dir = paths.concat(data_root, 'exp/proposals/train'),  -- exp/proposals/train, exp/DT_proposals/formatted/train
    val_anno_file = paths.concat(data_root, 'exp/anno/val.t7'),
    val_img_dir = paths.concat(data_root, 'Data/VID/val'),
    val_prop_dir = paths.concat(data_root, 'exp/proposals/val'),  -- exp/proposals/val, exp/DT_proposals/formatted/val
  }
  
  -- model specific params
  if model == 'stmn' then
    opt.model = 'stmn'
    opt.load_mode = 'rfcn_STMM_W'
    opt.retrain = paths.concat(data_root, 'models/rfcn.t7')
  elseif model == 'rfcn' then
    opt.model = 'rfcn'
    opt.load_mode = 'full'
    opt.retrain = 'no'
  end

  -- params for static image dataset
  local static_opt = 
    { 
      data_root = static_root,
      proposal_dir = paths.concat(static_root, 'exp/proposals'),
      dataset = 'imagenet',
      year = '2014',
      proposals = 'sharpmask',  -- 'sharpmask', 'selective_search'
      train_min_proposal_size = 0,
      train_min_gtroi_size = 0,
      best_proposals_number = 1000,
      sample_n_per_box = 30,
      sample_sigma = 20,
      train_set = opt.imagenet_det_subset, 
      images_per_batch = opt.DET_images_per_batch,
      scale = opt.scale,  -- 600
      max_size = opt.max_size,  -- 1000
      nDonkeys = opt.nDonkeys,
      scale_jitter = opt.scale_jitter,  -- uniformly jitter the scale by this frac
      aspect_jitter = opt.aspect_jitter,  -- uniformly jitter the scale by this frac
      crop_likelihood = opt.crop_likelihood,
      brightness_var = opt.brightness_var,  -- data augmentation, default 0.4 
      contrast_var = opt.contrast_var,  -- default 0.4 
      saturation_var = opt.saturation_var,  -- default 0.4  
      lighting_var = opt.lighting_var,  -- default 0.4 
      fg_threshold = opt.fg_threshold,     -- if -1, then set to bg_threshold_max
      bg_threshold_min = opt.bg_threshold_min,  -- 0.1 for Fast-RCNN, whereas 0 for Faster-RCNN
      bg_threshold_max = opt.bg_threshold_max,
      batchSize = opt.batch_size,
      fg_fraction = opt.fg_fraction,
      manualSeed = opt.manualSeed,
      transformer = opt.transformer,
      fg_threshold = opt.fg_threshold,     
      bg_threshold_min = opt.bg_threshold_min,  
      bg_threshold_max = opt.bg_threshold_max,
    }

  opt = xlua.envparams(opt)
  return opt, static_opt
end


function config.get_eval_config(model, model_path, data_root, ckpt_root)
  local vis_root = paths.concat(ckpt_root, 'stats')
  local model_root = paths.concat(ckpt_root, 'models')

  -- general opts
  local opt = {
     dataset = 'ImageNetVID',
     anno_file = data_root .. '/exp/anno/val.t7', -- set to val or test
     img_dir = data_root .. '/Data/VID/val',  -- '/Data/VID/val', '/Data/VID/val_deblur'
     prop_dir = data_root .. '/exp/proposals/val',  -- '/exp/proposals/val', '/exp/DT_proposals/formatted/val'
     test_nms_threshold = 1.0,  -- default to 0.3, 1.0
     score_threshold = 1e-5,  -- default to 0.001, 1e-5
     seq_per_batch = 1,  -- This should always be set to 1 during evaluation
     timestep_per_batch = 11, -- default to 11
     eval_timestep_per_batch = 11,  -- default to 11
     frame_stride = 10,  -- default to 10
     mem_dim = 512,
     nGPU = 1,
     test_batch_size = 1000,  -- set to nil if use all proposals
     scale = 360,  -- default to 360
     max_size = 600,  -- default to 600
     save_mat_file = true, -- default to false
     eval_res_dir = paths.concat(ckpt_root, 'eval'), 
     retrain = model_path,
     manualSeed = 777,
  }
  
  -- model specific params
  opt.model = model

  opt = xlua.envparams(opt)
  return opt
end

return config













