import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math

def match_flow_warp_kv(kv, flow, video_length, is_batch=False, first=None):
    bs, f, d, c = kv.shape
    h, w = flow.shape[-2:]
    flow_divisor = int((h*w) // d )
    flow_divisor = int(math.sqrt(flow_divisor))
    target_resolution = (int(h // flow_divisor), int(w // flow_divisor))
    if flow_divisor in [4, 2, 1]:
        mask = flow[:, 2:3, ...]
        mask = F.interpolate(mask, target_resolution, mode='bilinear')
        attn_masks = mask.reshape(f, -1) > 0.5 # 0是黑色unvis，1是白色vis
        if is_batch:
            attn_masks = torch.logical_not(attn_masks)  # In-place operation
            kv = kv[:, attn_masks]
            first = first[:, 0, ...]
            kv = torch.cat([first, kv], dim=1)
            kv = repeat(kv, "b d c -> b f d c", f=video_length)
        else:
            attn_masks = torch.cat([attn_masks[0:1, ...], ~attn_masks[1:]], dim=0)
            kv = repeat(kv[:, attn_masks], "b d c -> b f d c", f=video_length)
        return kv
    else:
        raise ValueError('The resolution of the input tensor is not supported.')


def rgb_flow_warp_kv(kv, flow, video_length, previous=None):
    bs, f, d, c = kv.shape
    h, w = flow.shape[-2:]
    flow_divisor = int(math.sqrt((h*w) // d))
    target_resolution = (h // flow_divisor, w // flow_divisor)

    if flow_divisor in [2, 1]:
        mask = F.interpolate(flow[:, 2:3, ...], target_resolution, mode='bilinear')
        kv = rearrange(kv, 'b f d c -> b f c d').reshape(bs, f, c, *target_resolution)
        flows = F.interpolate(flow[:, :2, ...] / flow_divisor, target_resolution, mode='bilinear')

        if previous is not None:
            previous = rearrange(previous, 'b f d c -> b f c d').reshape(bs, 1, c, *target_resolution)

        if bs == 2:
            kv[0, ...] = warp_accss(kv[0, ...], flows, mask=mask, video_length=video_length, previous=previous[0, ...] if previous is not None else None)
            kv[1, ...] = warp_accss(kv[1, ...], flows, mask=mask, video_length=video_length, previous=previous[1, ...] if previous is not None else None)
        else:
            kv[0, ...] = warp_accss(kv[0, ...], flows, mask=mask, video_length=video_length, previous=previous)

        kv = rearrange(kv.reshape(bs, f, c, d), 'b f c d -> b f d c')

    elif flow_divisor == 4:
        mask = F.interpolate(flow[:, 2:3, ...], target_resolution, mode='bilinear')
        kv = repeat(kv[:, mask.reshape(f, -1) > 0.5], "b d c -> b f d c", f=video_length)

    else:
        raise ValueError('The resolution of the input tensor is not supported.')

    return kv


def rgb_flow_warp_nose(kv, flow, video_length, query_previous=None):
    bs, f, d, c = kv.shape
    h, w = flow.shape[-2:]
    flow_divisor = int(math.sqrt((h*w) // d))
    target_resolution = (int(h // flow_divisor), int(w // flow_divisor))

    if flow_divisor in [4, 2, 1]:
        flows = F.interpolate(flow[:, :2, ...] / flow_divisor, target_resolution, mode='bilinear')
        mask = F.interpolate(flow[:, 2:3, ...], target_resolution, mode='bilinear')
        kv = rearrange(kv, 'b f d c -> b f c d').reshape(bs, f, c, *target_resolution)

        if query_previous is not None:
            query_previous = rearrange(query_previous, 'b f d c -> b f c d').reshape(bs, c, *target_resolution)

        if bs == 1:
            kv[0, ...] = warp_acc(kv[0, ...], flows, mask=mask, video_length=video_length, query_previous=query_previous)
        else:
            kv[0, ...] = warp_acc(kv[0, ...], flows, mask=mask, video_length=video_length, query_previous=query_previous[0, ...] if query_previous is not None else None)
            kv[1, ...] = warp_acc(kv[1, ...], flows, mask=mask, video_length=video_length, query_previous=query_previous[1, ...] if query_previous is not None else None)

        kv = rearrange(kv.reshape(bs, f, c, d), 'b f c d -> (b f) d c')

    else:
        raise ValueError('The resolution of the input tensor is not supported.')

    return kv


def warp_acc(tenInput, tenFlow, mask=None, video_length=None, query_previous=None):
    backwarp_tenGrid = {}

    device = tenInput.device
    flow_type = tenInput.dtype

    former_frame_index = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(video_length - 1)), dim=0)
    tenInput_warp = tenInput[former_frame_index, ...]
    if query_previous is not None:
        tenInput_warp[0, ...] = query_previous

    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k].to(flow_type) + tenFlow)  # bs_f, resH, resW, 2
    g = g.permute(0, 2, 3, 1)#.contiguous()

    tenInput_warp = F.grid_sample(input=tenInput_warp, grid=g, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)
    output = tenInput_warp * mask + (1 - mask) * tenInput
    # output = tenInput_warp * mask  # 保留未遮挡区域

    # del backwarp_tenGrid, g, tenInput, tenInput_warp
    # torch.cuda.empty_cache()

    return output


def warp_accss(tenInput, tenFlow, mask=None, video_length=None, previous=None):
    backwarp_tenGrid = {}

    device = tenInput.device
    flow_type = tenInput.dtype

    tenInput_warp = tenInput

    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k].to(flow_type) + tenFlow)  # bs_f, resH, resW, 2
    g = g.permute(0, 2, 3, 1)  # .contiguous()


    for i in range(video_length):
        if previous is not None:   # 这么写是为了区分是否使用batch_infer的方式
            if i == 0:
                tenInput_warp[0:1] = F.grid_sample(input=previous, grid=g[0:1], mode='bilinear', padding_mode='zeros',
                                                align_corners=True)
                tenInput[0:1] = tenInput_warp[0:1] * mask[0:1] + (1 - mask[0:1]) * tenInput[0:1]
            else:
                tenInput_warp[i:i + 1] = F.grid_sample(input=tenInput_warp[i - 1:i], grid=g[i:i + 1], mode='bilinear',
                                                    padding_mode='zeros',
                                                    align_corners=True)
                tenInput[i:i + 1] = tenInput_warp[i:i + 1] * mask[i:i + 1] + (1 - mask[i:i + 1]) * tenInput[i:i + 1]
        else:
            if i > 0:  # 第i帧，用上一帧和这一帧的后向flow进行warp
                tenInput_warp[i:i + 1] = F.grid_sample(input=tenInput_warp[i - 1:i], grid=g[i:i + 1], mode='bilinear',
                                                    padding_mode='zeros',
                                                    align_corners=True)
                tenInput[i:i + 1] = tenInput_warp[i:i + 1] * mask[i:i + 1] + (1 - mask[i:i + 1]) * tenInput[i:i + 1]

    output = tenInput

    # del backwarp_tenGrid, g, tenInput, tenInput_warp
    # torch.cuda.empty_cache()

    return output


