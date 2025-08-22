import gc
import inspect
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.utils.checkpoint

from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import flow_to_image, save_image

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from deps.gmflow.gmflow.gmflow import GMFlow
from infer_util.flow_utils import get_flow_and_occ
from infer_util.infer_util import read_video, read_video_wh
from infer_util.util import backup_profile, save_videos_grid, set_logger, save_tensor_images_folder

from hack_attn.attnen_store import register_attention_control_v2
from hack_attn.controller import AttentionControl_v2

from annotator.util import get_control, HWC3
import logging

from libs.piplines import VideoControlNetPipeline
from libs.unet import UNet3DConditionModel
# from libs.controlnet3d import ControlNetModel
from libs.controlnet import ControlNetModel3D


prompt_type = 'fresco_text'
if prompt_type == 'null_text':
    POS_PROMPT = ""
    NEG_PROMPT = ""
elif prompt_type == 'fresco_text':
    POS_PROMPT = " ,best quality, extremely detailed"
    NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
elif prompt_type == 'controlvideo_text':
    POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
    NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(
        pretrained_model_path: str,
        output_dir: str,
        pretrained_controlnet_path: str,
        validation_data: Dict,
        control_config: Dict,
        seed: Optional[int] = None,
        **kwargs
):

    output_dir_log = output_dir
    os.makedirs(output_dir_log, exist_ok=True)

    *_, config = inspect.getargvalues(inspect.currentframe())

    backup_profile(config, output_dir_log)
    set_logger(output_dir_log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logging.info(output_dir_log)

    weight_dtype = torch.float16
    # prepare models
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(dtype=weight_dtype)
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder=control_config.unet_subfolder).to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(pretrained_controlnet_path).to(dtype=weight_dtype)
    apply_control = get_control(control_config.type)
    flownet = GMFlow(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        attention_type='swin',
        ffn_dim_expansion=4,
        num_transformer_layers=6,
    ).to(device)
    checkpoint = torch.load('deps/models/gmflow_sintel-0c07dcb3.pth',
                            map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flownet.load_state_dict(weights, strict=False)
    flownet.eval()

    # prepare VideoControlNetPipeline
    validation_pipeline = VideoControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
        scheduler=scheduler)
    validation_pipeline.enable_vae_slicing()
    validation_pipeline.enable_xformers_memory_efficient_attention()
    validation_pipeline.to(device)

    # no grad
    for param in flownet.parameters():
        param.requires_grad = False
    for param in validation_pipeline.unet.parameters():
        param.requires_grad = False
    for param in validation_pipeline.controlnet.parameters():
        param.requires_grad = False
    
    # prepare data
    video, con = read_video_wh(video_path=validation_data.video_path, video_length=validation_data.video_length,
                            width=validation_data.width, height=validation_data.height,
                            frame_rate=validation_data.sample_frame_rate, start_frame=validation_data.sample_start_idx)
    
    # prepare condition data
    pil_annotation = []
    for i in con:
        i = i.cpu().numpy()
        if control_config.type == 'canny':
            anno = apply_control(i, control_config.low_threshold, control_config.high_threshold)
        else:
            anno = apply_control(i)
        anno = HWC3(anno)
        anno = Image.fromarray(anno)
        pil_annotation.append(anno)

    # stack control with all frames with shape [b c f h w]
    control = np.stack(pil_annotation)
    control = np.array(control).astype(np.float32) / 255.0   # control在0-1之间
    control = torch.from_numpy(control).to(device)
    control = control.unsqueeze(0)  # [f h w c] -> [b f h w c ]
    control = rearrange(control, "b f h w c -> b c f h w")
    control = control.to(weight_dtype)
    #  for save
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    original_pixels = (original_pixels + 1.0) / 2.0  # -1,1 -> 0,1
    origin_save = original_pixels.cpu().float()
    control_save = control.cpu().float()

    # prepare rgb flow
    video_lengths, _, height, width  = video.shape 
    logging.info(f"processing video length {video_lengths}, height {height}, width {width}")
    validation_data.height, validation_data.width = height, width  # update height and width
    flow_frame = rearrange(con.to(device).float(), 'f h w c -> f c h w') # f h w c - f c h w

    # v2   batch_interpreter
    controller = AttentionControl_v2()
    if controller is not None:
        controller.set_total_step(validation_data.num_steps)  # 设置step
    register_attention_control_v2(unet, controller)

    # 设置noise一致性
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)   # set seed
    if validation_data.start == 'noise':
        shape = (1, 4, 1, int(validation_data.height//8), int(validation_data.width//8))  # b c f h w 
        noise = torch.randn(shape, device=device, generator=generator, dtype=weight_dtype)  # 这个是对的

    # 进行batch式的infer, 每个batch是8帧
    set_batch_max_length = 8
    indices = [list(range(i, min(i+set_batch_max_length, video_lengths))) for i in range(0, video_lengths, set_batch_max_length)]

    torch.cuda.empty_cache()
    # 开始infer
    unet.eval()
    controlnet.eval()
    batch_ind = 0
    samples = []  # sotre all-batch samples
    for batch_ind, indice in enumerate(indices):

        controller.batch_id = batch_ind
        if batch_ind == 0:
            controller.set_task('init_first')
        else:
            controller.set_task('next_batch')

        validation_data.video_length = len(indice)  # 进行batch操作
        logging.info(f"processing batch {batch_ind}, clip length {validation_data.video_length}")

        # condition_image
        control_i = control[:, :, indice, :, :]  # b c f h w
        # flow
        former_frame_index = [i - 1 for i in indice]
        if batch_ind == 0:
            former_frame_index[0] = 0  # 第一个batch的第一帧，设置为0
        target_frame = flow_frame[indice, ...]  # f c h w
        source_frame = flow_frame[former_frame_index, ...]
        with torch.no_grad():
            if_pixel_occ = False  # generate occlusion mask form (pixel-wise threshold and flow-based occlusion)
            bwd_occ, bwd_flow = get_flow_and_occ(flow_model=flownet, image1=source_frame, image2=target_frame, if_pixel_occ=if_pixel_occ)

            # vis
            # y = flow_to_image(bwd_occ)
            # y = y.float() / 255
            # save_image(bwd_occ, "bwd_occ.png")
            # z = flow_to_image(bwd_flow.float())
            # z = z.float() / 255
            # save_image(z, "bwd_flow.png")

            if not if_pixel_occ:
                # 模糊mask
                blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
                blend_mask_pre = blur(F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
                # 模糊等级
                # blur = T.GaussianBlur(kernel_size=(7, 7), sigma=(15, 15))  # 减小高斯模糊的内核大小和sigma值
                # blend_mask_pre = blur(F.max_pool2d(bwd_occ, kernel_size=7, stride=1, padding=3)) 
                blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ, 0, 1)
                mask = 1 - F.max_pool2d(blend_mask_pre, kernel_size=8)
            else:
                # 精细mask
                mask = 1 - F.interpolate(bwd_occ, scale_factor=1. / 8, mode='bilinear')

            bwd_flow = F.interpolate(bwd_flow / 8.0, scale_factor=1. / 8, mode='bilinear')
            save_image(mask, "flow_three_c.png")
            flow_i = torch.cat((bwd_flow.to(weight_dtype), mask.to(weight_dtype)), dim=1)
            del bwd_occ, bwd_flow, blend_mask_pre, mask

        ddim_inv_latent = repeat(noise, 'b c 1 h w -> b c f h w', f=validation_data.video_length)
        # 进行frame-frame
        gc.collect()
        torch.cuda.empty_cache()

        sample_batch = []   # 为了多个prompt
        if validation_data.edit_type == 'DDIM':
            for idx, prompt in enumerate(validation_data.prompts):
                prompts = [prompt + POS_PROMPT]
                n_prompts = [NEG_PROMPT]
                if batch_ind == 0:
                    logging.info(f"processing prompt {prompt}")
                infer_sample = validation_pipeline(prompt = prompts,
                                                   generator=generator, latents=ddim_inv_latent,
                                                   image=control_i, negative_prompt=n_prompts,
                                                   controlnet_conditioning_scale=control_config.control_scale,
                                                   flow=flow_i, controller=controller, **validation_data).videos
                sample_batch.append(infer_sample)
        samples.append(sample_batch[0])  # 这一步会增加显存, 搬到cpu上

        gc.collect()
        torch.cuda.empty_cache()

    logging.info("save translated video")
    samples = torch.cat(samples, dim=2)   # b c f h w 
    # videos
    saves = [origin_save, control_save, samples]
    save_path = f"{output_dir}/{prompt}/batch.mp4"
    saves = rearrange(saves, 'm b c t h w -> (m b) c t h w')
    save_videos_grid(saves, save_path)

    # Save result images
    logging.info("save translated images")
    save_path1 = f"{output_dir}/{prompt}/result"
    save_tensor_images_folder(samples, save_path1)
    save_path = f"{output_dir}/{prompt}/{'origal'}"
    save_tensor_images_folder(origin_save, save_path)
    save_path_cond = f"{output_dir}/{prompt}/{'condition'}"
    save_tensor_images_folder(control_save, save_path_cond)