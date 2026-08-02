import os
import torch
import numpy as np
from PIL import Image


def save_image_tensor(image_tensor, output_path="output/"):
    # Replace NaN/Inf, clamp to [0,1], then convert [B,C,H,W] -> [H,W,C] uint8
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    img = img.clamp(0.0, 1.0).detach().cpu()
    ar = (img[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(ar).save(output_path)
