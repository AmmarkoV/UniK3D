import gradio as gr
import numpy as np
import torch
from PIL import Image
import cv2
import os
import tempfile
from unik3d.models import UniK3D
from unik3d.utils.camera import Fisheye624, Spherical, OPENCV, Pinhole, MEI

port="7860"
server_name="0.0.0.0"

# === Load model once ===
model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
model.resolution_level = 9
model.interpolation_mode = "bilinear"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# === Main inference function ===
def predict_depth_and_distance(image, dist_min, dist_max):
    # Convert image to tensor
    rgb = np.array(image.convert("RGB"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model.infer(rgb=rgb_torch[0], camera=None, normalize=True, rays=None)

    # Process depth map
    depth = outputs["depth"].squeeze().cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    d_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    #depth_16bit = (d_norm * 65535.0).astype(np.uint16)
    #depth_img = Image.fromarray(depth_16bit)
    depth_8bit = (d_norm * 255.0).astype(np.uint8)
    #depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_16bit, alpha=0.03), cv2.COLORMAP_JET)
    depth_img = Image.fromarray(depth_8bit)

    # Process distance map (optional)
    distance = outputs.get("distance", None)
    if distance is not None:
        distance_np = distance.squeeze().cpu().numpy()
        print("Distance Min : ",distance_np.min()," Distance Max: ",distance_np.max())
        #dist_min = 0
        #dist_max = 20
        dist_norm = (distance_np - dist_min) / (dist_max - dist_min + 1e-8)
        #distance_16bit = (dist_norm * 65535.0).astype(np.uint16) 
        #distance_img = Image.fromarray(distance_16bit)
        distance_8bit = (dist_norm * 255.0).astype(np.uint8)
        distance_img = Image.fromarray(distance_8bit)
    else:
        distance_img = Image.new("RGB", depth.shape[::-1], color=(0, 0, 0))

    return depth_img, distance_img

# === Gradio Interface ===
demo = gr.Interface(
    fn=predict_depth_and_distance,
    inputs=[
        gr.Image(type="pil", label="Upload RGB Image"),
        gr.Number(value=0, label="Distance Min (meters)"),
        gr.Number(value=20, label="Distance Max (meters)")
    ],
    outputs=[
        gr.Image(type="pil", label="Depth Map (8-bit normalized)"),
        gr.Image(type="pil", label="Distance Map (8-bit normalized)")
    ],
    title="UniK3D Depth and Distance Estimator",
    description="Upload an RGB image to get depth and distance maps using UniK3D. You can adjust the min/max distance range for visualization."
)

if __name__ == "__main__":
    demo.launch(server_name=server_name, server_port=int(port))

