import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
import cv2
import numpy as np
import random
import torch

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_video(video_path, video_length, width=512, height=512, frame_rate=None):
    vr = decord.VideoReader(video_path)  
    if frame_rate is None:
        frame_rate = max(1, len(vr) // video_length)
    # frame_rate = len(vr) // video_length
    sample_index = list(range(0, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)
    con = video
    video = rearrange(video, "f h w c -> f c h w")
    video = (video / 127.5 - 1.0)
    return video, con

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def calculate_dimensions(H, W, width):
    H = float(H)
    W = float(W)
    k = float(width) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    return H, W

def read_video_wh(video_path, video_length, width=512, height=512, frame_rate=None, start_frame=0):
    vr0 = decord.VideoReader(video_path)
    fps = vr0.get_avg_fps()
    H, W = vr0.get_batch([0]).shape[1:3] # 1 h w c
    H, W = calculate_dimensions(H, W, width)
    
    vr = decord.VideoReader(video_path, width=W, height=H)
    if frame_rate is None:
        frame_rate = max(1, len(vr) // video_length)
    sample_index = list(range(start_frame, len(vr), frame_rate))[:video_length]
    video = vr.get_batch(sample_index)
    con = video
    video = rearrange(video, "f h w c -> f c h w")
    video = (video / 127.5 - 1.0)
    
    return video, con