-- This is a useful module which, can be used for debugging tunnel

local DbTunnel, parent = torch.class('nn.DbTunnel', 'nn.Module')

function DbTunnel:__init(db_id)
   parent.__init(self)
   self.gradInput = nil
   self.db_id = db_id or '0'
   self.print_db_id = false
end

function DbTunnel:updateOutput(input)
  if self.print_db_id then
    print(string.format('DbTunnel: updateOutput, db_id = %s', self.db_id))
  end
  self.output = input
  return self.output
end

function DbTunnel:updateGradInput(input, gradOutput)
  if self.print_db_id then
    print(string.format('DbTunnel: updateGradInput, db_id = %s', self.db_id))
  end
  self.gradInput = gradOutput
  return self.gradInput
end

function DbTunnel:accGradParameters(input, gradOutput, scale)
  if self.print_db_id then
    print(string.format('DbTunnel: accGradParameters, db_id = %s', self.db_id))
  end
  local db_stop = true
  assert(db_stop)
end