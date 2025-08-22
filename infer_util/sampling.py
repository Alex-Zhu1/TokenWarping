from util import ddim_inversion
from einops import rearrange, repeat
import torch

# POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
POS_PROMPT = " ,best quality, extremely detailed"
# POS_PROMPT = ""
# NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# NEG_PROMPT = ""
# NEG_PROMPT = "ugly, blurry, low res, unrealistic, unaesthetic"

def VideoGen(validation_data, generator, latents, validation_pipeline, ddim_inv_scheduler, train_data, control, weight_dtype, control_scale, samples, flow):
    torch.cuda.empty_cache()
    if validation_data.start == 'noise':
        B, C, f, H, W = latents.shape
        noise = torch.randn([C, H, W], device=latents.device, generator=generator)  # generator 一开始失真是由于acclerate的seed_everything。但是diffusion有一些随机是不应该被固定的
        noise = rearrange(noise, 'c h w -> 1 c h w')
        noise = repeat(noise, '1 ... -> f ...', f=f)
        noise = rearrange(noise, 'f c h w -> 1 f c h w')
        noise = repeat(noise, '1 ... -> b ...', b=B)
        noise = rearrange(noise, "b f c h w -> b c f h w")
        ddim_inv_latent = noise.to(weight_dtype)
    elif validation_data.start == 'inversion':
        ddim_inv_latent = ddim_inversion(
            validation_pipeline, ddim_inv_scheduler, 
            video_latent=latents,
            num_inv_steps=validation_data.num_steps, 
            prompt=train_data.prompt, controls=control,
            controlnet_conditioning_scale=control_scale, 
            flow=flow)[-1].to(weight_dtype)
    else:
        raise ValueError(f"Unknown start type {validation_data.start}")

    if validation_data.edit_type == 'DDIM':
        for idx, prompt in enumerate(validation_data.prompts):
            sample = validation_pipeline(prompt + POS_PROMPT, 
                                         generator=generator, 
                                         latents=ddim_inv_latent,     
                                         image=control,
                                         controlnet_conditioning_scale=control_scale, 
                                         flow=flow,negative_prompt=NEG_PROMPT,
                                         **validation_data).videos
            samples.append(sample)
    else:
        raise ValueError(f"Unknown edit type {validation_data.edit_type}")

    return samples

