
local AdaLinearScale, parent = torch.class('nn.AdaLinearScale', 'nn.Module')

function AdaLinearScale:__init(std_multiplier)
   parent.__init(self)
   self.gradInput = nil
   self.std_multiplier = std_multiplier
   self.upper_bound = 1
   self.clamp = nn.Clamp(0, self.upper_bound)
   self.multiplier = nn.MulConstant(1.0 / self.upper_bound, true)
   self.net = nn.Sequential()
                :add(self.clamp)
                :add(self.multiplier)
end

function AdaLinearScale:setUpperBound()
  assert(self.upper_bound > 0, 'upper_bound cannot be less or equal to 0.')
  self.clamp.max_val = self.upper_bound
  self.multiplier.constant_scalar = 1.0/self.upper_bound
end


function AdaLinearScale:updateOutput(input)
  self.output = self.output or input.new()
  self.output:resizeAs(input)
  local input_flat = input:view(-1)
  local pos_input = input_flat[input_flat:gt(0)]
  if pos_input:nElement() > 0 then
    local std = torch.std(pos_input)
    -- ONLY FOR DEBUGGING
    --local db_mean = torch.mean(pos_input)
    --local db_meansub_input = pos_input - db_mean
    --local db_sum_variance = torch.sum(torch.cmul(db_meansub_input, db_meansub_input))
    --local db_std = torch.sqrt(db_sum_variance / pos_input:nElement()) 
    local mean = torch.mean(pos_input)
    if tostring(std) == 'nan' or tostring(mean) == 'nan' then
      self.upper_bound = 1.0
    else
      self.upper_bound = mean + std * self.std_multiplier
    end
  else
    self.upper_bound = 1.0
  end
  
  --if DB_GLOBAL_FLAG then
  --  self.count = self.count or 0
  --  self.count = self.count + 1
  --  self.upper_bound_sum = self.upper_bound_sum or 0
  --  self.upper_bound_sum = self.upper_bound_sum + self.upper_bound
  --  print(string.format('%g', self.upper_bound_sum/self.count))
  --end 
  
  ---- precaution to avoid infs
  --if self.upper_bound > 1e10 or self.upper_bound < -1e10 then
  --  self.upper_bound = 1.0
  --end
  
  self:setUpperBound()
  local output = self.net:forward(input)
  self.output:copy(output)
  return self.output
end

function AdaLinearScale:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or input.new()
  self.gradInput:resizeAs(input)
  local gradInput = self.net:backward(input, gradOutput)
  self.gradInput:copy(gradInput)
  return self.gradInput
end

