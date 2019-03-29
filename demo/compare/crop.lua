require 'image'

cmd = torch.CmdLine()
cmd:option('-apath',        '/dataset/SR_compare',              'Absolute path')
cmd:option('-set',          'Set5',                             'Test set')
cmd:option('-name',         'baby',                             'Test image')
cmd:option('-scales',       '4',                                'Scales')
cmd:option('-id',           1,                                  'Id of cropped image')
cmd:option('-lt',           '1_1',                              'left, top')
cmd:option('-wh',           '1_1',                              'width, height')
cmd:option('-ps',           '.',                                'square patch size')
cmd:option('-works',        'Aplus+SRCNN+VDSR+SRResNet+SRResNet_reproduce+Ours_Single+Ours_Multi',        'works to compare')

local opt = cmd:parse(arg or {})

local apath = opt.apath
local works = opt.works:split('+')
local set = opt.set
local imgName = opt.name
local scales = opt.scales:split('_')
for i, sc in ipairs(scales) do
    scales[i] = tonumber(sc)
end
local id = opt.id
local lt = opt.lt:split('_')
local wh = opt.wh:split('_')
local patchSize = opt.ps
if patchSize ~= '.' then
    patchSize = tonumber(patchSize)
    for i = 1, 2 do
        lt[i] = tonumber(lt[i])
        wh[i] = patchSize
    end
else
    for i = 1, 2 do
        lt[i] = tonumber(lt[i])
        wh[i] = tonumber(wh[i])
    end
    assert(wh[1] * wh[2] ~= 1)
end

local w, h = wh[1], wh[2]
local left, top = lt[1], lt[2]
local right, bottom = left + w - 1, top + h - 1

local savePath = 'cropped'
if not paths.dirp(savePath) then
    paths.mkdir(savePath)
end

local ext = '.png'
id = id .. '_' .. w .. 'x' .. h

local function save(work, scale, img)
    local name = set .. '_' .. imgName .. '_' .. work .. '_sc=' .. scale
    name = name .. '_x=' .. lt[1] .. '_y=' .. lt[2] .. '_ps=' .. patchSize
    image.save(paths.concat(savePath, name .. ext), img)
end

for _,scale in pairs(scales) do
    local ilr = image.load(paths.concat(apath, 'Bicubic', set, 'X' .. scale, imgName .. ext))
    ilr = ilr[{{}, {top, bottom}, {left, right}}]
    save('Bicubic', scale, ilr)
    -- local ilr_name = paths.concat('cropped', set .. '_' .. imgName .. '_' .. id .. '_Bicubic_x' .. scale .. ext)
    -- image.save(ilr_name, ilr)

    for _,work in pairs(works) do
		if not ((work:find('SRResNet') and scale ~= 4) or (work == 'SRResNet' and (set == 'val' or set == 'Urban100'))) then
        -- if (not (work == 'SRResNet' and (scale ~= 4 or set == 'val'))) and (work == 'SRResNet_reproduce' and scale ~= 4) then
            local sr
            if work == 'SRResNet' then
                sr = image.load(paths.concat(apath, work, set, 'X' .. scale, imgName .. '_SRResNet-MSE' .. ext))
            else
                sr = image.load(paths.concat(apath, work, set, 'X' .. scale, imgName .. ext))
            end
            sr = sr[{{}, {top, bottom}, {left, right}}]
            save(work, scale, sr)
            -- local sr_name = paths.concat('cropped', set .. '_' .. imgName .. '_' .. id .. '_' .. work .. '_x' .. scale .. ext)
            -- image.save(sr_name, sr)
        end
    end
end

local hr = image.load(paths.concat(apath, 'GT', set, imgName .. ext))
local hr_ = hr [{{}, {top, bottom}, {left, right}}]
save('GT', 0, hr_)
-- local hr_name = paths.concat('cropped', set .. '_' .. imgName .. '_' .. id .. '_HR' .. ext)
-- image.save(hr_name, hr)

image.drawRect(hr, left, top, right, bottom, {lineWidth = 7, color={255,255,0}, inplace=true})
save('Box', 0, hr)
