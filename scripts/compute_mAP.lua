package.path = package.path .. ';../?.lua'

require 'torch'
require 'cunn'
require 'paths'
local mu = require '../myutils'
local utils = require '../utils'

local this_file_dir = paths.dirname(paths.thisfile(nil))

local res_dir = '/data/fanyix/VID/dataset/RFCN_STMN/eval'
local write_dir = '/data/fanyix/VID/dataset/RFCN_STMN/eval'
local use_mat_file = 'false'
local root_dir = paths.concat(this_file_dir, '..', '..')
local iter_array = torch.range(1, 1, 1):totable()

local matlab_loc = 'matlab'
local dataset_root = paths.concat(root_dir, 'dataset/')
local eval_data_dir = paths.concat(root_dir, 'multipathnet', 'external/VID_devkit_data/')
local eval_script_dir = paths.concat(root_dir, 'multipathnet', 'external/VID_devkit/evaluation/')

for _, iter in ipairs(iter_array) do
 local exec_cmd = 
      string.format('%s -nodesktop -nodisplay -r "cd %s; eval_VID(%d, \'%s\', \'%s\', \'%s\', \'%s\', \'%s\');exit;"', 
      matlab_loc, eval_script_dir, iter, res_dir, dataset_root, eval_data_dir, write_dir, use_mat_file)
 --print(exec_cmd)
 print('Start quant evaluation ...')
 local handle = io.popen(exec_cmd)
 local result = handle:read("*a")
 handle:close()
 print('Quant evaluation done.')
end
