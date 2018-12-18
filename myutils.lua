
local cjson = require 'cjson'
local torch_utils = {} 

function torch_utils.wd_sgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) weight decay with single or individual parameters
   if torch.norm(dfdx) > 1e-8 then
     if wd ~= 0 then
          dfdx:add(wd, x)
     elseif wds then
        if not state.decayParameters then
           state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end
        state.decayParameters:copy(wds):cmul(x)
        dfdx:add(state.decayParameters)
     end
  
     -- (3) apply momentum
     if mom ~= 0 then
        if not state.dfdx then
           state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
        else
           state.dfdx:mul(mom):add(1-damp, dfdx)
        end
        if nesterov then
           dfdx:add(mom, state.dfdx)
        else
           dfdx = state.dfdx
        end
     end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)

   -- (5) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      x:add(-clr, dfdx)
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end

function torch_utils.dir(directory)
  local list = paths.dir(directory)
  local new_list = {}
  for i = 1, #list do
    if (list[i] ~= '.' and list[i] ~= '..') then
      table.insert(new_list, list[i])
    end 
  end
  return new_list
end


-- see if the file exists
function torch_utils.file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function torch_utils.strip_postfix(filename)
  local splits = torch_utils.split(filename, '%.')
  local num = #splits
  local clean_filename
  if num > 1 then
    clean_filename = table.concat({table.unpack(splits, 1, num-1)})
  else
    clean_filename = filename
  end
  return clean_filename
end

function torch_utils.safe_concat(data, dim) 
  local count = 0
  for item_idx, item in ipairs(data) do
    count = count + item:size(dim)
  end
  local data_size = data[1]:size()
  data_size[dim] = count
  local data_cat = data[1].new():resize(data_size)
  local count = 1
  for _, item in ipairs(data) do
    data_cat:narrow(dim, count, item:size(dim)):copy(item)
    count = count + item:size(dim)
  end
  return data_cat
end

function torch_utils.VOCap(rec, prec)
  local mrec = rec.new(rec:nElement()+2)
  mrec[1] = 0
  mrec[mrec:nElement()] = 1
  mrec:narrow(1, 2, mrec:nElement()-2):copy(rec)
  
  local mpre = prec.new(prec:nElement()+2)
  mpre[1] = 0
  mpre[mpre:nElement()] = 0
  mpre:narrow(1, 2, mpre:nElement()-2):copy(prec)
  
  local ap = 0
  local N = mpre:nElement()
  for _, i in ipairs(torch.range(N-1, 1, -1):totable()) do
    mpre[i] = math.max(mpre[i], mpre[i+1])
  end
  
  local idx = torch.nonzero((mrec[{{2, N}}]-mrec[{{1, N-1}}]):ne(0))
  if idx:nElement() > 0 then
    idx = idx:view(-1) + 1
    idx = idx:long()
    ap = torch.sum(torch.cmul((mrec:index(1, idx) - mrec:index(1, idx-1)), mpre:index(1, idx)))
  end
  return ap
end

function torch_utils.vector_unique(input_tensor)
  input_tensor = input_tensor:contiguous()
  local unique_elements = {} --tracking down all unique elements
  local output_table = {} --result table/vector
  input_tensor = input_tensor:view(-1)
  for idx = 1, input_tensor:nElement() do
    local val = input_tensor[idx]
    unique_elements[val] = idx
  end
  return unique_elements
end


function torch_utils.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end


function torch_utils.read_lines_into_tensor(file)
  if not torch_utils.file_exists(file) then return {} end
  local count, dim = 0, 0
  for line in io.lines(file) do
    if count == 0 then
      dim = #torch_utils.split(line, ' ')
    end 
    count = count + 1
  end
  local tensor = torch.FloatTensor(count, dim):zero()
  local count = 1
  for line in io.lines(file) do
    local splits = torch_utils.split(line, ' ')
    for str_idx, str in ipairs(splits) do
      tensor[{count, str_idx}] = tonumber(str)
    end
    count = count + 1
  end
  return tensor
end

function torch_utils.sample_from_weights(weights)
  local norm_weights = torch.div(weights, torch.sum(weights) + 1e-10)
  norm_weights = norm_weights:view(-1)
  local cum_weights = torch.cumsum(norm_weights)
  local sample_val = torch.rand(1)[1] - 1e-8
  local ge_idx = torch.nonzero(cum_weights:ge(sample_val)):view(-1)
  ge_idx = ge_idx[1]
  return ge_idx
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function torch_utils.lines_from(file)
  if not torch_utils.file_exists(file) then return {} end
  local lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end


function torch_utils.size_string(tensor_input)
  assert(torch.isTensor(tensor_input))
  local tensor_size = tensor_input:size()
  local s = ''
  for i = 1, #tensor_size do
    if s ~= '' then 
      s = s .. ',' .. string.format('%d', tensor_size[i])
    else
      s = string.format('%d', tensor_size[i])
    end
  end
  return s
end

function torch_utils.print_net(net)
  for i,module in ipairs(net:listModules()) do
    print(module)
  end
end

function torch_utils.keys(obj)
  local keyset={}
  local n=0
  for k,v in pairs(obj) do
    n=n+1
    keyset[n]=k
  end
  return keyset
end

function torch_utils.charTensor2String(tensor)
  local str_table = {}
  for idx = 1, tensor:nElement() do
    table.insert(str_table, string.format('%c', tensor[idx]))
  end
  str_table = table.concat(str_table)
  return str_table
end

--[[
Utility function to check that a Tensor has a specific shape.

Inputs:
- x: A Tensor object
- dims: A list of integers
--]]
function torch_utils.check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    local msg = 'Expected %d, got %d'
    assert(x:size(i) == d, string.format(msg, d, x:size(i)))
  end
end


function torch_utils.get_kwarg(kwargs, name, default)
  if kwargs == nil then kwargs = {} end
  if kwargs[name] == nil and default == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif kwargs[name] == nil then
    return default
  else
    return kwargs[name]
  end
end


function torch_utils.get_size(obj)
  local size = 0
  for k, v in pairs(obj) do size = size + 1 end
  return size
end


function torch_utils.read_json(path)
  local f = io.open(path, 'r')
  local s = f:read('*all')
  f:close()
  return cjson.decode(s)
end


function torch_utils.write_json(path, obj)
  local s = cjson.encode(obj)
  local f = io.open(path, 'w')
  f:write(s)
  f:close()
end

function torch_utils.db_checkpoint(iter)
  local db_stop = 0
  if iter % 10 == 0 then
    db_stop = 1
  end
  if iter % 100 == 0 then
    db_stop = 1
  end 
  if iter % 30 == 0 then
    db_stop = 1
  end
  if iter % 300 == 0 then
    db_stop = 1
  end
  return db_stop
end

function torch_utils.init_recorder(T)
  local recorder = {smoothed_loss_arr = {}, raw_loss_arr = {}, loss_iter_arr = {}, ptr = 1, T = T}
  return recorder
end

function torch_utils.retrieve_loss(struct, start_round)
  --start_round = start_round or 1
  --local loss, iter = {}, {}
  --if #struct.smoothed_loss_arr >= start_round then
  --  loss = {table.unpack(struct.smoothed_loss_arr, start_round, #struct.smoothed_loss_arr)}
  --  iter = {table.unpack(struct.loss_iter_arr, start_round, #struct.loss_iter_arr)}
  --end
  local loss, iter = struct.smoothed_loss_arr, struct.loss_iter_arr
  return loss, iter
end

function torch_utils.update_loss(struct, loss, iter)
  local raw_loss_arr = struct.raw_loss_arr
  local smoothed_loss_arr = struct.smoothed_loss_arr
  local loss_iter_arr = struct.loss_iter_arr
  local T = struct.T
  local ptr = struct.ptr
  local smoothed_loss
  if #smoothed_loss_arr > 0 then
    smoothed_loss = smoothed_loss_arr[#smoothed_loss_arr]
  else
    smoothed_loss = 0
  end
  
  local cur_len = #raw_loss_arr
  if cur_len < T then
    smoothed_loss = (smoothed_loss * cur_len + loss) / (cur_len + 1)
  else
    smoothed_loss = smoothed_loss + (loss - raw_loss_arr[ptr]) / T
  end
  raw_loss_arr[ptr] = loss
  ptr = ptr % T + 1
  table.insert(smoothed_loss_arr, smoothed_loss)
  table.insert(loss_iter_arr, iter)
  
  struct.ptr = ptr
  struct.raw_loss_arr = raw_loss_arr
  struct.smoothed_loss_arr = smoothed_loss_arr
  struct.loss_iter_arr = loss_iter_arr
  return struct
end

function torch_utils.init_html_container(rows, cols)
  local im_paths, captions = {}, {}
  for row_idx = 1, rows do
    im_paths[row_idx] = {}
    captions[row_idx] = {}
    for col_idx = 1, cols do
      im_paths[row_idx][col_idx] = {}
      captions[row_idx][col_idx] = {}
    end
  end
  return im_paths, captions
end

-- Determine the longest prefix among a list of strings
-- This function is borrowed from facebook utils at:
-- https://github.com/facebook/fblualib/blob/master/fblualib/util/fb/util/init.lua
function torch_utils.longest_common_prefix(strings)
    if #strings == 0 then
        return ''
    end
    local prefix = strings[1]
    for i = 2, #strings do
        local s = strings[i]
        local len = 0
        for j = 1, math.min(#s, #prefix) do
            if s:sub(j, j) == prefix:sub(j, j) then
                len = len + 1
            else
                break
            end
        end
        prefix = prefix:sub(1, len)
        if len == 0 then
            break
        end
    end
    return prefix
end

function torch_utils.relative_path(ref_path, target_path)
  local common_prefix = torch_utils.longest_common_prefix({ref_path, target_path})
  local _, end_str_ptr = string.find(target_path, common_prefix)
  local rel_path = string.sub(target_path, end_str_ptr + 1)
  return rel_path
end

function torch_utils.write_html(filename, im_paths, captions, height, width)
    paths.mkdir(paths.dirname(filename))
    local f = io.open(filename, 'w')
    io.output(f)
    io.write('<!DOCTYPE html>\n')
    io.write('<html><body>\n')
    io.write('<table>\n')
    for row_idx, captions_row in ipairs(captions) do
        io.write('<tr>\n')
        for col_idx, col in ipairs(captions_row) do
            io.write('<td>')
            io.write(col)
            io.write('</td>')
            io.write('    ')
        end
        io.write('\n</tr>\n')
        io.write('<tr>\n')
        for col_idx, col in ipairs(im_paths[row_idx]) do
            io.write('<td><img src="')
            io.write(col)
            io.write(string.format('" height=%d width=%d"/></td>', height, width))
            io.write('    ')
        end
        io.write('\n</tr>\n')
        io.write('<p></p>')
    end
    io.write('</table>\n')
    io.close()
end

-- Compatibility: Lua-5.1
function torch_utils.split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
         table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

return torch_utils


