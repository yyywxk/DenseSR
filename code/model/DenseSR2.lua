require 'nn'
require 'model/common'

local function createModel(opt)

    --growth rate
   local growthRate = opt.growthRate

   --dropout rate, set it to 0 to disable dropout, non-zero number to enable dropout and set drop rate
   local dropRate = opt.dropRate

   --# channels before entering the first Dense-Block
   local nChannels = 2 * growthRate

   --compression rate at transition layers
   local reduction = opt.reduction

   --whether to use bottleneck structures
   local bottleneck = opt.bottleneck

   --N: # dense connected layers in each denseblock
   local N = (opt.depth - 4)/3
   if bottleneck then N = N/2 end


   function addLayer(model, nChannels, opt)
      if opt.optMemory >= 3 then
         model:add(nn.DenseConnectLayerCustom(nChannels, opt))
      else
         model:add(nn.Concat(2)
            :add(nn.Identity())
            :add(DenseConnectLayerStandard(nChannels, opt))) 
			--	
      end
   end


   function addTransition(model, nChannels, nOutChannels, opt, last, pool_size)
      if opt.optMemory >= 3 then     
         model:add(nn.JoinTable(2))
      end

      --model:add(cudnn.SpatialBatchNormalization(nChannels))
      --model:add(cudnn.ReLU(true))      
      if last then
         model:add(cudnn.SpatialAveragePooling(pool_size, pool_size))
         model:add(nn.Reshape(nChannels))      
      else
         model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
         if opt.dropRate > 0 then model:add(nn.Dropout(opt.dropRate)) end
         --model:add(cudnn.SpatialAveragePooling(2, 2))
		 
      end      
   end


   local function addDenseBlock(model, nChannels, opt, N)
      for i = 1, N do 
         addLayer(model, nChannels, opt)
         nChannels = nChannels + opt.growthRate
      end
      return nChannels
   end
	
	local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval

    local body = seq()
	--for i = 1, opt.nResBlock do
        --body:add(resBlock(opt.nFeat, false, actParams, opt.scaleRes, opt.ipMulc))
    --end
	--number of layers in each block
    if opt.depth == 121 then
       stages = {6, 12, 24, 16}
    elseif opt.depth == 169 then
       stages = {6, 12, 32, 32}
    elseif opt.depth == 201 then
       stages = {6, 12, 48, 32}
    elseif opt.depth == 161 then
       stages = {6, 12, 36, 24}
    else
       stages = {opt.d1, opt.d2, opt.d3, opt.d4}
    end
	
	--Initial transforms follow ResNet(224x224)
    --body:add(cudnn.SpatialConvolution(64, nChannels, 3,3, 1,1, 1,1))
	body:add(cudnn.SpatialConvolution(256, nChannels, 3,3, 1,1, 1,1))
    --body:add(cudnn.SpatialBatchNormalization(nChannels))
    --body:add(cudnn.ReLU(true))
    --body:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

    --Dense-Block 1 and transition (56x56)
    nChannels = addDenseBlock(body, nChannels, opt, stages[1])
    addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)

    --Dense-Block 2 and transition (28x28)
    nChannels = addDenseBlock(body, nChannels, opt, stages[2])
    addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)

    --Dense-Block 3 and transition (14x14)
    nChannels = addDenseBlock(body, nChannels, opt, stages[3])
    addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)

    --Dense-Block 4 and transition (7x7)
    nChannels = addDenseBlock(body, nChannels, opt, stages[4])
    --addTransition(body, nChannels, nChannels, opt, true, 7)
	-- addTransition(body, nChannels, nChannels, opt)
	addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)
	
	--Dense-Block 1 and transition (56x56)
    nChannels = addDenseBlock(body, nChannels, opt, stages[1])
    addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)

    --Dense-Block 2 and transition (28x28)
    nChannels = addDenseBlock(body, nChannels, opt, stages[2])
    addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)

    --Dense-Block 3 and transition (14x14)
    nChannels = addDenseBlock(body, nChannels, opt, stages[3])
    addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)

    --Dense-Block 4 and transition (7x7)
    nChannels = addDenseBlock(body, nChannels, opt, stages[4])
    --addTransition(body, nChannels, nChannels, opt, true, 7)
	-- addTransition(body, nChannels, nChannels, opt)
	addTransition(body, nChannels, math.floor(nChannels*reduction), opt)
    nChannels = math.floor(nChannels*reduction)
	
		
    --body:add(conv(nChannels,64, 3,3, 1,1, 1,1))
	body:add(conv(nChannels,256, 3,3, 1,1, 1,1))
	
	

    --ret = seq():add(conv(opt.nChannel,64, 3,3, 1,1, 1,1))
	ret = seq():add(conv(opt.nChannel,256, 3,3, 1,1, 1,1))
	--ret:add(mulc(opt.scaleRes, opt.ipMulc))--set ipMulc into true and set scaleRes as 0.1 		
	--local ret = nn.Sequential()
    if opt.globalSkip then
        ret:add(addSkip(body, true))
    else
        ret:add(body)
    end
	ret:add(upsample_wo_act(opt.scale[1], opt.upsample, 256))
		--add(upsample_wo_act(opt.scale[1], opt.upsample, 256))
       :add(conv(256,opt.nChannel, 3,3, 1,1, 1,1))
	   --:add(conv(256,opt.nChannel, 3,3, 1,1, 1,1))
	    
    return ret
	
end

return createModel
