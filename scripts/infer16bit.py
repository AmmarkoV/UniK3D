import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import (MEI, OPENCV, BatchCamera, Fisheye624, Pinhole, Spherical)
from unik3d.utils.visualization import colorize, grayscale, grayscale_flipped, save_file_ply

def save(rgb, outputs, name, base_path, save_map=False, save_pointcloud=False):
    os.makedirs(base_path, exist_ok=True)
    depth  = outputs["depth"]
    rays   = outputs["rays"]
    points = outputs["points"]

    depth = depth.cpu().numpy()  # shape: (1, H, W)
    import cv2

    if save_map:
        # Original colorized depth for visualization
        #Image.fromarray(colorize(depth.squeeze())).save(os.path.join(base_path, f"{name}_depth.png"))

        # === Save 16-bit depth ===
        d = depth.squeeze()  # shape (H, W)
        d_min = d.min()
        d_max = d.max()
        if d_max - d_min < 1e-6:
            d_max = d_min + 1e-3
        d_norm = (d - d_min) / (d_max - d_min)
        d_16bit = (d_norm * 65535.0).astype(np.uint16)
        depth16_path = os.path.join(base_path, f"{name}_depth.png")
        cv2.imwrite(depth16_path, d_16bit)
        print(f"Saved 16-bit depth map: {depth16_path}")

        # Save rays as image
        #rays = ((rays + 1) * 127.5).clip(0, 255)
        #Image.fromarray(rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()).save(os.path.join(base_path, f"{name}_rays.png"))

    if save_pointcloud:
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))

def infer(model, args, input_file):
  try:
    rgb = np.array(Image.open(input_file).convert("RGB"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    camera = None
    if args.camera_path is not None:
        with open(args.camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)

    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)
    name = os.path.splitext(os.path.basename(input_file))[0]
    save(
        rgb_torch,
        outputs,
        name=name,
        base_path=args.output,
        save_map=args.save,
        save_pointcloud=args.save_ply,
    )
  except:
    print("Failed processing ",input_file)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser( description="Training script", conflict_handler="resolve" )
    parser.add_argument("--input", type=str, required=True, help="Path to input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--config-file", type=str, required=True, default="./configs/eval/vitl.json", help="Path to config file. Please check ./configs/eval.", )
    parser.add_argument( "--camera-path", type=str, default=None, help="Path to camera parameters json file. See assets/demo for a few examples.", )
    parser.add_argument( "--save", action="store_true", help="Save outputs as (colorized) png." )
    parser.add_argument( "--save-ply", action="store_true", help="Save pointcloud as ply." )
    parser.add_argument( "--resolution-level", type=int, default=9, help="Resolution level in [0,10). Higher values increases details but decreases speed.", choices=list(range(10)), )
    parser.add_argument( "--interpolation-mode", type=str, default="bilinear", help="Output interpolation.", choices=["nearest", "nearest-exact", "bilinear"], )
    args = parser.parse_args()

    print("Torch version:", torch.__version__)
    version = args.config_file.split("/")[-1].split(".")[0]
    name = f"unik3d-{version}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()



    # === File loader: single image or txt file ===
    if os.path.isfile(args.input):
        if args.input.endswith(".txt"):
            with open(args.input, "r") as f:
                files = f.read().splitlines()
        else:
            files = [args.input]
    else:
        raise FileNotFoundError(f"Input path '{args.input}' not found")

    for idx, input_file in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Processing: {input_file}")
        infer(model, args, input_file)

    #infer(model, args)
