
-- script to create a annotation file for Torch7 for ImageNet VID
package.path = package.path .. ';../?.lua'

require 'torch'
require 'paths'
local myutils = require 'myutils'

-- re-organize data
local anno = {}
local json_dir = '/data/fanyi-data/ILSVRC2015/Annotations/VID/val/json'
local t7_file = '/data/fanyi-data/ILSVRC2015/Annotations/VID/val/anno.t7'
local img_id_file = '/data/fanyi-data/ILSVRC2015/ImageSets/VID/val.txt'

-- read data from json
for file in paths.iterfiles(json_dir) do
  local json_file = paths.concat(json_dir, file)
  local res = myutils.read_json(json_file)
  for _, vid in ipairs(res) do
    local video_name = vid.video_name
    anno[video_name] = {}
    
    -- get img size: dirty hack to handle bad json conversion
    local img_size_tensor = {}
    if torch.type(vid.im_size[1]) == 'table' then
      for _, frm in ipairs(vid.im_size) do
        table.insert(img_size_tensor, torch.FloatTensor(frm))
      end
    else
      table.insert(img_size_tensor, torch.FloatTensor(vid.im_size))
    end
    img_size_tensor = torch.cat(img_size_tensor, 2):permute(2, 1)
    
    -- collect frame name
    local fr_name = {}
    for _, frame_name in ipairs(vid.im_list) do
      assert(torch.type(frame_name[1][1]) == 'string')
      table.insert(fr_name, frame_name[1][1])
    end
    
    local obj_arr = {}
    for _, obj in ipairs(vid.obj) do
      local boxes_tensor = {}
      
      -- dirty hack to handle bad json conversion
      if obj.boxes == nil then
        assert(obj[2] == nil and obj[1][2] == nil)
        obj = obj[1][1]
      end
      
      -- dirty hack to handle bad json conversion
      if torch.type(obj.boxes[1]) == 'table' then
        for _, frm in ipairs(obj.boxes) do
          table.insert(boxes_tensor, torch.FloatTensor(frm))
        end
      else
        table.insert(boxes_tensor, torch.FloatTensor(obj.boxes))
      end
    
      boxes_tensor = torch.cat(boxes_tensor, 2):permute(2, 1)      
      obj.boxes = boxes_tensor
      table.insert(obj_arr, obj)
    end
    anno[video_name].obj = obj_arr
    
    anno[video_name].im_list = fr_name
    anno[video_name].im_size = img_size_tensor    
    print(string.format('%g-th video processed', #myutils.keys(anno)))
  end
end


-- Break different segments into different 'obj'
for vid_name, vid in pairs(anno) do
  local sub_obj_arr = {}
  for _, obj in ipairs(vid.obj) do
    
    local obj_cat = obj.category
    local obj_idx = obj.obj_idx
    local boxes = obj.boxes
    local frame_id = boxes[{{}, 2}]
    local n = frame_id:nElement()
    
    if n <= 1 then
      table.insert(sub_obj_arr, obj)
    else
      local segs = frame_id[{{2, n}}] - frame_id[{{1, n-1}}]
      local endpoints = torch.nonzero(segs:ne(1))
      if endpoints:nElement() > 0 then
        endpoints = torch.cat({endpoints.new({0}), endpoints:view(-1), endpoints.new({n})}, 1)
      else
        endpoints = torch.cat({endpoints.new({0}), endpoints.new({n})}, 1)
      end
      
      -- create all objs
      for pp = 1, endpoints:nElement()-1 do
        local sub_obj = {}
        local start_idx = endpoints[pp] + 1
        local end_idx = endpoints[pp + 1]
        local sub_boxes = boxes[{{start_idx, end_idx}, {}}]:clone()
        sub_obj.category = obj_cat
        sub_obj.obj_idx = obj_idx
        sub_obj.start_frame = boxes[{start_idx, 2}]
        sub_obj.end_frame = boxes[{end_idx, 2}]
        sub_obj.boxes = sub_boxes
        table.insert(sub_obj_arr, sub_obj)
      end  
    end  
  end
  vid.obj = sub_obj_arr
end

-- Load an image list to get the unique image id for each and every frame
local frame_name_to_idx = {}
for line in io.lines(img_id_file) do 
  local loc = string.find(line,' ')
  local frame_name = string.sub(line, 1, loc-1) .. '.JPEG'
  local frame_idx = tonumber(string.sub(line, loc+1))
  frame_name_to_idx[frame_name] = frame_idx
end

for vid_name, vid in pairs(anno) do
  local global_idx = {}
  for im_idx, im in ipairs(vid.im_list) do 
    local key = vid_name .. '/' .. im
    local value = frame_name_to_idx[key]
    assert(value ~= nil)
    global_idx[im_idx] = value
  end
  vid.global_idx = torch.IntTensor(global_idx)
end

torch.save(t7_file, anno)


















