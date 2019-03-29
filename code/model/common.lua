require 'nn'
require 'cudnn'
require 'cunn'

--------------------------------------------------------------------------
-- Below are list of common modules used in various architectures.
-- Thoses are defined as global variables in order to make other codes uncluttered.
--------------------------------------------------------------------------
seq = nn.Sequential
conv = nn.SpatialConvolution
deconv = nn.SpatialFullConvolution
relu = nn.ReLU
prelu = nn.PReLU
rrelu = nn.RReLU
elu = nn.ELU
leakyrelu = nn.LeakyReLU
bnorm = nn.SpatialBatchNormalization
avgpool = nn.SpatialAveragePooling
shuffle = nn.PixelShuffle
pad = nn.Padding
concat = nn.ConcatTable
id = nn.Identity
cadd = nn.CAddTable
join = nn.JoinTable
mulc = nn.MulConstant

--select activation function
function act(actParams, nOutputPlane)
    local nOutputPlane = actParams.nFeat or nOutputPlane
    local type = actParams.actType

    if type == 'relu' then
        return relu(true)
    elseif type == 'prelu' then
        return prelu(nOutputPlane)
    elseif type == 'rrelu' then
        return rrelu(actParams.l, actParams.u, true)
    elseif type == 'elu' then
        return elu(actParams.alpha, true)
    elseif type == 'leakyrelu' then
        return leakyrelu(actParams.negval, true)
    else
        error('unknown activation function!')
    end
end

function addSkip(model, global)

    local model = seq()
        :add(concat()
            :add(model)
            :add(id()))
        :add(cadd(true))

    -- global skip or local skip connection of residual block
    model:get(2).global = global or false

    return model
end

function upsample(scale, method, nFeat, actParams)
    local scale = scale or 2
    local method = method or 'espcnn'
    local nFeat = nFeat or 64

    local actType = actParams.actType
    local l, u = actParams.l, actParams.u
    local alpha, negval = actParams.alpha, actParams.negval
    actParams.nFeat = nFeat

    local model = seq()
    if method == 'deconv' then
        if scale == 2 then
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(act(actParams))
        elseif scale == 3 then
            model:add(deconv(nFeat,nFeat, 9,9, 3,3, 3,3))
            model:add(act(actParams))
        elseif scale == 4 then
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(act(actParams))
            model:add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
            model:add(act(actParams))
        end
    elseif method == 'espcnn' then  -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if scale == 2 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(act(actParams))
        elseif scale == 3 then
            model:add(conv(nFeat,9*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(3))
            model:add(act(actParams))
        elseif scale == 4 then
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(act(actParams))
            model:add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
            model:add(shuffle(2))
            model:add(act(actParams))
        end
    end

    return model
end

function upsample_wo_act(scale, method, nFeat)
    local scale = scale or 2
    local method = method or 'espcnn'
    local nFeat = nFeat or 64

    if method == 'deconv' then
        if scale == 2 then
            return deconv(nFeat,nFeat, 6,6, 2,2, 2,2)
        elseif scale == 3 then
            return deconv(nFeat,nFeat, 9,9, 3,3, 3,3)
        elseif scale == 4 then
            return seq()
                :add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
                :add(deconv(nFeat,nFeat, 6,6, 2,2, 2,2))
        elseif scale == 1 then
            return id()
        end
    elseif method == 'espcnn' then  -- Shi et al., 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
        if scale == 2 then
            return seq()
                :add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(2))
        elseif scale == 3 then
            return seq()
                :add(conv(nFeat,9*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(3))
        elseif scale == 4 then
            return seq()
                :add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(2))
                :add(conv(nFeat,4*nFeat, 3,3, 1,1, 1,1))
                :add(shuffle(2))
        elseif scale == 1 then
            return id()
        end
    end
end

function resBlock(nFeat, addBN, actParams, scaleRes, ipMulc)
    local nFeat = nFeat or 64
    local scaleRes = (scaleRes and scaleRes ~= 1) and scaleRes or false
	local ipMulc = ipMulc or false
	if not scaleRes then
		assert(not ipMulc, 'Please specify -scaleRes option')
	end

    actParams.nFeat = nFeat

    if addBN then
        return addSkip(seq()
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat))
            :add(act(actParams))
            :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
            :add(bnorm(nFeat)))
    else
        if scaleRes then 
            return addSkip(seq()
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
                :add(act(actParams))
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
                :add(mulc(scaleRes, ipMulc)))
        else
            return addSkip(seq()
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
                :add(act(actParams))
                :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1)))
        end
    end
end

function cbrcb(nFeat, addBN, actParams)
    local nFeat = nFeat or 64
    actParams.nFeat = nFeat

    return seq()
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
end

function crc(nFeat, actParams)
    local nFeat = nFeat or 64
    actParams.nFeat = nFeat

    return seq()
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
end

function brcbrc(nFeat, actParams)
    return seq()
        :add(bnorm(nFeat))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
        :add(bnorm(nFeat))
        :add(act(actParams))
        :add(conv(nFeat,nFeat, 3,3, 1,1, 1,1))
end

local MultiSkipAdd, parent = torch.class('nn.MultiSkipAdd', 'nn.Module')

function MultiSkipAdd:__init(ip)
    parent.__init(self)
    self.inplace = ip
end

--This function takes the input like {Skip, {Output1, Output2, ...}}
--and returns {Output1 + Skip, Output2 + Skip, ...}
--It also supports in-place calculation
function MultiSkipAdd:updateOutput(input)
    self.output = {}

    if self.inplace then
        for i = 1, #input[2] do
            self.output[i] = input[2][i]
        end
    else
        for i = 1, #input[2] do
            self.output[i] = input[2][i]:clone()
        end
    end

    for i = 1, #input[2] do
        self.output[i]:add(input[1])
    end
    
    return self.output
end

function MultiSkipAdd:updateGradInput(input, gradOutput)
    self.gradInput = {gradOutput[1]:clone():fill(0), {}}

    if self.inplace then
        for i = 1, #input[2] do
            self.gradInput[1]:add(gradOutput[i])
            self.gradInput[2][i] = gradOutput[i]
        end
    else
        for i = 1, #input[2] do
            self.gradInput[1]:add(gradOutput[i])
            self.gradInput[2][i] = gradOutput[i]:clone()
        end
    end

    return self.gradInput
end

--DenseSR 

local function ShareGradInput(module, key)
   assert(key)
   module.__shareGradInputKey = key
   return module
end

--------------------------------------------------------------------------------
-- Standard densely connected layer (memory inefficient)
-- BN-ReLU-Conv(1*1)-BN-ReLU-Conv(3*3)
--------------------------------------------------------------------------------
function DenseConnectLayerStandard(nChannels, opt)
   local net = nn.Sequential()

   -- net:add(ShareGradInput(cudnn.SpatialBatchNormalization(nChannels), 'first'))
   -- net:add(cudnn.ReLU(true))   
   if opt.bottleneck then
      net:add(ShareGradInput(cudnn.SpatialConvolution(nChannels, 4 * opt.growthRate, 3, 3, 1, 1, 1, 1), 'first'))
      nChannels = 4 * opt.growthRate
      if opt.dropRate > 0 then net:add(nn.Dropout(opt.dropRate)) end
      --net:add(cudnn.SpatialBatchNormalization(nChannels))
      net:add(cudnn.ReLU(true))      
   end
   -- return addSkip(seq()
                -- :add(ShareGradInput(cudnn.SpatialConvolution(nChannels, 4 * opt.growthRate, 3, 3, 1, 1, 1, 1), 'first'))
                -- :add(cudnn.ReLU(true)) 
                -- :add(cudnn.SpatialConvolution(4 * opt.growthRate, nChannels, 3, 3, 1, 1, 1, 1)))
   
   net:add(cudnn.SpatialConvolution(nChannels, opt.growthRate, 3, 3, 1, 1, 1, 1))
   
   --net:add(cudnn.SpatialConvolution(nChannels, opt.growthRate, 3, 3, 1, 1, 1, 1))
   if opt.dropRate > 0 then net:add(nn.Dropout(opt.dropRate)) end

   return net
end

--------------------------------------------------------------------------------
-- Customized densely connected layer (memory efficient)
-- net1:BN-ReLU  net2:Conv(1*1)-BN-ReLU-Conv(3*3)
--------------------------------------------------------------------------------
local DenseConnectLayerCustom, parent = torch.class('nn.DenseConnectLayerCustom', 'nn.Container')

function DenseConnectLayerCustom:__init(nChannels, opt)
   parent.__init(self)
   self.train = true
   self.opt = opt

   self.net1 = nn.Sequential()
   self.net1:add(ShareGradInput(cudnn.SpatialBatchNormalization(nChannels), 'first'))
   self.net1:add(cudnn.ReLU(true))  

   self.net2 = nn.Sequential()
   if opt.bottleneck then
      self.net2:add(cudnn.SpatialConvolution(nChannels, 4*opt.growthRate, 1, 1, 1, 1, 0, 0))
      nChannels = 4 * opt.growthRate
      self.net2:add(cudnn.SpatialBatchNormalization(nChannels))
      self.net2:add(cudnn.ReLU(true))
   end
   self.net2:add(cudnn.SpatialConvolution(nChannels, opt.growthRate, 3, 3, 1, 1, 1, 1))

   -- contiguous outputs of previous layers
   self.input_c = torch.Tensor():type(opt.tensorType) 
   -- save a copy of BatchNorm statistics before forwarding it for the second time when optMemory=4
   self.saved_bn_running_mean = torch.Tensor():type(opt.tensorType)
   self.saved_bn_running_var = torch.Tensor():type(opt.tensorType)

   self.gradInput = {}
   self.output = {}

   self.modules = {self.net1, self.net2}
end

function DenseConnectLayerCustom:updateOutput(input)

   if type(input) ~= 'table' then
      self.output[1] = input
      self.output[2] = self.net2:forward(self.net1:forward(input))
   else
      for i = 1, #input do
         self.output[i] = input[i]
      end
      torch.cat(self.input_c, input, 2)
      self.net1:forward(self.input_c)
      self.output[#input+1] = self.net2:forward(self.net1.output)
   end

   if self.opt.optMemory == 4 then
      local running_mean, running_var = self.net1:get(1).running_mean, self.net1:get(1).running_var
      self.saved_bn_running_mean:resizeAs(running_mean):copy(running_mean)
      self.saved_bn_running_var:resizeAs(running_var):copy(running_var)
   end

   return self.output
end

function DenseConnectLayerCustom:updateGradInput(input, gradOutput)

   if type(input) ~= 'table' then
      self.gradInput = gradOutput[1]
      if self.opt.optMemory == 4 then self.net1:forward(input) end
      self.net2:updateGradInput(self.net1.output, gradOutput[2])
      self.gradInput:add(self.net1:updateGradInput(input, self.net2.gradInput))
   else
      torch.cat(self.input_c, input, 2)
      if self.opt.optMemory == 4 then self.net1:forward(self.input_c) end
      self.net2:updateGradInput(self.net1.output, gradOutput[#gradOutput])
      self.net1:updateGradInput(self.input_c, self.net2.gradInput)
      local nC = 1
      for i = 1, #input do
         self.gradInput[i] = gradOutput[i]
         self.gradInput[i]:add(self.net1.gradInput:narrow(2,nC,input[i]:size(2)))
         nC = nC + input[i]:size(2)
      end
   end

   if self.opt.optMemory == 4 then
      self.net1:get(1).running_mean:copy(self.saved_bn_running_mean)
      self.net1:get(1).running_var:copy(self.saved_bn_running_var)
   end

   return self.gradInput
end

function DenseConnectLayerCustom:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.net2:accGradParameters(self.net1.output, gradOutput[#gradOutput], scale)
   if type(input) ~= 'table' then
      self.net1:accGradParameters(input, self.net2.gradInput, scale)
   else
      self.net1:accGradParameters(self.input_c, self.net2.gradInput, scale)
   end
end

function DenseConnectLayerCustom:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local lastNext = '   `-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'DenseConnectLayerCustom'
   str = str .. ' {' .. line .. tab .. '{input}'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. lastNext .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. '{output}'
   str = str .. line .. '}'
   return str
end
