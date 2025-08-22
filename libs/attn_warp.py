import torch
import torch.nn.functional as F
from einops import rearrange




def warp_acc(tenInput, tenFlow, mask=None, video_length=None, mask_value=0):
    backwarp_tenGrid = {}
    device = tenInput.device
    flow_type = tenInput.dtype

    former_frame_index = torch.arange(video_length) - 1
    former_frame_index[0] = 0
    tenInput_warp = tenInput[former_frame_index, ...]

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

    g = (backwarp_tenGrid[k].to(flow_type) + tenFlow).permute(0, 2, 3, 1)  # bs_f, resH, resW, 2
    if tenInput.shape[-1] in [32, 64]:
        tenInput_warp = F.grid_sample(input=tenInput_warp, grid=g, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)
        output = tenInput * (1 - mask) + tenInput_warp * mask
        output = torch.cat((tenInput[0:1, ...], output[1:, ...]))
    return output


def warp(tenInput, tenFlow, mask, mask_value=0):
    backwarp_tenGrid = {}
    k = (str(tenFlow.device), str(tenFlow.size()))
    device = tenInput.device
    flow_type = tenInput.dtype
    b, _, h, w = tenFlow.shape
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, w, device=device).view(
            1, 1, 1, w).expand(b, -1, h, -1)
        tenVertical = torch.linspace(-1.0, 1.0, h, device=device).view(
            1, 1, h, 1).expand(b, -1, -1, w)
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)  # 设置了一个grid-----------+

    # tenFlow[:, 0:1, :, :] = 2 * tenFlow[:, 0:1, :, :] / (w - 1.0) - 1.0   @ in place operate
    # tenFlow[:, 1:2, :, :] = 2 * tenFlow[:, 1:2, :, :] / (w - 1.0) - 1.0
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k].to(flow_type) + tenFlow).permute(0, 2, 3, 1)
    output = F.grid_sample(input=tenInput, grid=g, mode='nearest', padding_mode='zeros', align_corners=True)
    if mask is not None:
        output = output * mask + (1 - mask) * mask_value
        # 前后景结合起来, 可以添加新语义
        if True:
            former_frame_index = torch.arange(b) - 1
            former_frame_index[0] = 0
            mask1 = mask[former_frame_index, ...]
            tenInput1 = tenInput * (1 - mask1) + mask1 * mask_value  # 将input的前景置为0，保留背景
            tenInput1 = tenInput1 * (1 - mask) + mask * mask_value  # 将warp后的前景的区域，在input中前景置为0
            output = output + tenInput1
            # 前后景，应该结合下面操作
            output = torch.cat((tenInput[0:1, ...], output[1:, ...]))  # 由于没有对x进行clone操作，导致这里第一帧前景置为0
        output = output.to(flow_type)
    # # 第一帧不需要warp
    return output


def kv_flow_warp(x, flow, mode='nearest', mask=None, mask_value=0):
    '''
    为啥写这个warp，是因为，img数值范围本身就是-1到1的，但是kv不是这个范围的
    Input:
        x: (bsz, c, h, w)
        flow: (bsz, 2, h, w)
        mask: (bsz, 1, h, w). 1 for valid region and 0 for invalid region. invalid region will be fill with "mask_value" in the output images.
    Output:
        y: (bsz, c, h, w)
    '''
    bsz, c, h, w = x.shape
    flow_type = x.dtype
    # mesh grid
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    stacks = [xx, yy]
    grid = torch.stack(stacks, dim=0).to(flow_type).requires_grad_(True)
    grid = grid[None].repeat(bsz, 1, 1, 1)
    grid = grid.to(x.device) + flow
    # scale to [-1, 1]
    x_grid = 2.0 * grid[:, 0] / (w - 1) - 1.0
    y_grid = 2.0 * grid[:, 1] / (h - 1) - 1.0
    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]
    output = F.grid_sample(x, grid, mode=mode, padding_mode='zeros', align_corners=True)
    # if mask is not None:
    #     output = torch.where(mask > 0.5, output, output.new_ones(1).mul_(mask_value))
    #     # 前后景结合起来
    #     former_frame_index = torch.arange(bsz) - 1
    #     former_frame_index[0] = 0
    #     mask1 = mask[former_frame_index, ...]   # 奇怪，这一步操作导致很大的抖动
    #     x1 = torch.where(mask1 < 0.5, x.clone().detach(), x.new_ones(1).mul_(mask_value))  # 将x中mask以内置0，外部不变 .clone().detach()
    #     x1 = torch.where(mask < 0.5, x1, x1.new_ones(1).mul_(mask_value))  # 将x1中mask以内置0，外部不变
    #     output = output + x1
    # # 第一帧不需要warp
    # outputs = torch.cat((x[0:1, ...], output[1:, ...]))  # 由于没有对x进行clone操作，导致这里第一帧前景置为0，但是效果更好
    return output


def flow_warps(kv, flow, video_length):
    resolution_mapping = {
        256: ((16, 16), 16.0),
        1024: ((32, 32), 8.0),
        4096: ((64, 64), 4.0)}  # 由于checkpoint里面不推荐全局变量引来的差异
    # 将flow加入训练，去掉clone和detach
    # interpolate是可回传的
    bs, f, d, c = kv.shape
    target_resolution, flow_divisor = resolution_mapping.get(d, (None, None))
    if target_resolution is not None:
        flows = flow[:, :2, ...]
        mask = flow[:, 2:3, ...]
        flows = F.interpolate(flows / flow_divisor, target_resolution, mode='bilinear')
        mask = F.interpolate(mask, target_resolution, mode='bilinear')
        kv = rearrange(kv, 'b f d c -> (b f) c d')
        bs_f = bs * f
        kv = kv.reshape(bs_f, c, *target_resolution)  # Reshape to the target resolution
        if bs_f == video_length:
            kv = warp_acc(kv, flows, mask=mask, video_length=video_length)
        else:
            kv[:video_length, ...] = warp_acc(kv[:video_length, ...], flows, mask=mask, video_length=video_length)
            kv[video_length:, ...] = warp_acc(kv[video_length:, ...], flows, mask=mask, video_length=video_length)
        kv = kv.reshape(bs_f, c, d).permute(0, 2, 1).contiguous()  # Reshape back to original shape
        kv = rearrange(kv, '(b f) d c -> b f d c', b=bs)
    return kv


def warp_acc_flow(x, flow, mode='nearest', mask=None, mask_value=0):
    '''
    warp an image/tensor according to given flow.
    Input:
        x: (bsz, c, h, w)
        flow: (bsz, c, h, w)
        mask: (bsz, 1, h, w). 1 for valid region and 0 for invalid region. invalid region will be fill with "mask_value" in the output images.
    Output:
        y: (bsz, c, h, w)
    '''
    bsz, c, h, w = x.size()
    flow_type = x.dtype
    # mesh grid
    xx = x.new_tensor(range(w)).view(1, -1).repeat(h, 1)
    yy = x.new_tensor(range(h)).view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(bsz, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(bsz, 1, 1, 1)
    grid = torch.cat((xx, yy), dim=1).to(flow_type)
    grid = grid + flow
    # scale to [-1, 1]
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(w - 1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(h - 1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)
    # 我们需要flow warp的过程是可微的，也就是flow和x有直接关系
    # Exclude the first frame from warping
    x_first_frame = x[0:1, ...]  # Extract the first frame
    x_remaining_frames = x[1:, ...]  # Extract the remaining frames
    grid = grid[1:, ...]
    output_remaining_frames = F.grid_sample(x_remaining_frames, grid, mode=mode, padding_mode='zeros', align_corners=True)
    # output = F.grid_sample(x.clone().detach(), grid, mode=mode, padding_mode='zeros', align_corners=True)
    output = torch.cat((x_first_frame, output_remaining_frames), dim=0)
    if mask is not None:
        output = torch.where(mask > 0.5, output, output.new_ones(1).mul_(mask_value))
        # 前后景结合起来
        former_frame_index = torch.arange(bsz) - 1
        former_frame_index[0] = 0
        mask1 = mask[former_frame_index, ...]
        x1 = torch.where(mask1 < 0.5, x.clone().detach(),
                         x.new_ones(1).mul_(mask_value))  # 将x中mask以内置0，外部不变 .clone().detach()
        x1 = torch.where(mask < 0.5, x1, x1.new_ones(1).mul_(mask_value))  # 将x1中mask以内置0，外部不变
        output = output + x1
        output = torch.cat((x[0:1, ...], output[1:, ...]))  # 由于没有对x进行clone操作，导致这里第一帧前景置为0，但是效果更好
    # # 第一帧不需要warp
    return output


def flow_warps_sim(sim, flow, video_length):
    resolution_mapping = {
        256: ((16, 16), 16.0),
        1024: ((32, 32), 8.0),
        4096: ((64, 64), 4.0)}  # 由于checkpoint里面不推荐全局变量引来的差异
    bs_f, h, d, c = sim.shape
    target_resolution, flow_divisor = resolution_mapping.get(d, (None, None))
    if target_resolution is not None:
        flows = flow[:, :2, ...]
        mask = flow[:, 3:4, ...]  # fg mask
        # flows = F.interpolate(flows / flow_divisor, target_resolution, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, target_resolution, mode='bilinear', align_corners=True)
        mask = mask[0:1, ...]  # 统一第一帧的fg mask
        mask = mask.unsqueeze(1)
        sim = sim.reshape(bs_f, h, d, *target_resolution)  # Reshape to the target resolution
        # 由于输入是batch_f, h, d, 64, 64,对每个通道都warp
        # 所以把head提取出来，h 个 batch_f d 64 64被flow bs_f 64 64 2 war
        if bs_f == video_length:
            # sim = warp_sim(sim, flows, mask=mask, video_length=video_length)
            sim = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
        else:
            # sim[:video_length, ...] = warp_sim(sim[:video_length, ...], flows, mask=mask, video_length=video_length)   #  这个是uncon的，是否不需要warp
            # sim[-video_length:, ...] = warp_sim(sim[-video_length:, ...], flows, mask=mask, video_length=video_length)
            sim[:video_length, ...] = sim[:video_length, ...] + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim[-video_length:, ...] = sim[-video_length:, ...] + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
        sim = sim.reshape(bs_f, h, d, c)
    return sim


def warp_sim(tenInput, tenFlow, mask=None, video_length=None):
    # save_image(mask, 'mask.jpg')
    bs_f, h, c, res_h, res_w = tenInput.shape
    backwarp_tenGrid = {}

    device = tenInput.device
    flow_type = tenInput.dtype

    # former_frame_index = torch.arange(video_length) - 1
    # former_frame_index[0] = 0
    # tenInput_warp = tenInput[former_frame_index, ...]
    tenInput_warp = tenInput[[0]*video_length, ...]

    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, res_w, device=device).view(1, 1, 1, res_w).expand(bs_f, -1, res_h, -1)
        tenVertical = torch.linspace(-1.0, 1.0, res_h, device=device).view(1, 1, res_h, 1).expand(bs_f, -1, -1, res_w)
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((res_w - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((res_h - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k].to(flow_type) + tenFlow).permute(0, 2, 3, 1)  # bs_f, resH, resW, 2

    if res_w in [32, 64]:
        # 维度转化下多出来的head
        tenInput_warp = rearrange(tenInput_warp, 'bsf h c resh resw ->(h bsf) c resh resw')
        g = g.unsqueeze(0).expand(h, -1, -1, -1, -1)
        g = rearrange(g, 'h bsf resh resw c -> (h bsf) resh resw c')
        mask = mask.unsqueeze(0).expand(h, -1, -1, -1, -1)
        mask = rearrange(mask, 'h bsf resh resw c -> (h bsf) resh resw c')
        tenInput_warp = F.grid_sample(input=tenInput_warp, grid=g, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)
        output = tenInput_warp + mask.masked_fill(mask == 0, torch.finfo(flow_type).min)
        output = rearrange(output, '(h bsf) c resh resw -> h bsf c resh resw', h=h)
        output = output.permute(1, 0, 2, 3, 4)
    else:
        output = tenInput
    return output
