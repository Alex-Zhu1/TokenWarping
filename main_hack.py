import os
from infer_util.util import available_devices, format_devices
device = available_devices(threshold=20000, n_devices=1)
device = [1]
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
from omegaconf import OmegaConf

# from infer_long import main  # flow is all frame inference
from infer_batch import main  # flow prediction is 8 frames batch-infer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--control_type",
        type=str,
        default='gold-fish',
        help="the type of control"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default='videos/gold-fish.mp4',
        help="the path to the input video"
    )

    parser.add_argument(
        "--source",
        type=str,
        default='A person is dancing',
        help="the prompt for source video"
    )

    parser.add_argument(
        "--target",
        type=str,
        default='watercolor gold fish',
        help="the prompt for target video"
    )

    parser.add_argument(
        "--unet_subfolder",
        type=str,
        default='unet',
        help="the subfolder for unet, rev, flat, moxin"
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default='outputs_7_5_output/',
        help="the path for saving"
    )

    parser.add_argument(
        "--max_step",
        type=int,
        default=300,
        help="the steps for training"
    )

    args = parser.parse_args()

    name = args.video_path.split('/')[-1]
    name = name.split('.')[0]
    config_root = "./configs/default/"
    config = os.path.join(config_root, f"{args.control_type}.yaml")
    para = OmegaConf.load(config)

    para.validation_data.video_path = args.video_path
    para.output_dir = os.path.join(args.out_root, f"{name}")

    para.validation_data.prompts = [args.target]  # 重写prompt
    para.control_config.unet_subfolder = args.unet_subfolder  # 重写lora模型

    main(**para)







