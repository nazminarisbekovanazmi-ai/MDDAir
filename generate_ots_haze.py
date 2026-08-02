"""
Synthesize OTS-style outdoor haze training data from clear images.

Input:  a directory of outdoor clear images (the 2061 OTS source images)
Output: hazy_train_ots/
            hazy/   -- synthesized hazy images  ({id}_{A:.2f}_{beta:.3f}.jpg)
            GT/     -- copies of the clear images ({id}.jpg)

The atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))

Depth proxy (no external depth model required): weighted blend of a
vertical gradient (sky=far, ground=close) and the image luminance
(brighter outdoor regions tend to be farther away), Gaussian-smoothed
to avoid sharp transitions.

Parameters sampled to match the SOTS-outdoor test range:
    A    ~ Uniform[0.7, 1.0]
    beta ~ Uniform[0.04, 0.20]
Run with --n_per_image 35 to produce ~72,135 pairs matching the paper.
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--clear_dir',   type=str,
                    default=r'C:\Users\Nazmi\Desktop\실험\Degradation_datasets\2061 OTS clear')
parser.add_argument('--out_dir',     type=str,
                    default=r'C:\Users\Nazmi\Desktop\실험\Degradation_datasets\hazy_train_ots')
parser.add_argument('--n_per_image', type=int, default=35)
parser.add_argument('--A_range',     type=float, nargs=2, default=[0.7, 1.0])
parser.add_argument('--beta_range',  type=float, nargs=2, default=[0.04, 0.20])
parser.add_argument('--depth_scale', type=float, default=15.0,
                    help='Scales normalized [0,1] depth to approximate metric depth (metres). '
                         'SOTS beta values assume metric depth, so without this scaling '
                         'haze is invisible. 15 gives realistic outdoor haze at beta 0.04-0.20.')
parser.add_argument('--seed',        type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
hazy_dir = os.path.join(args.out_dir, 'hazy')
gt_dir   = os.path.join(args.out_dir, 'GT')
os.makedirs(hazy_dir, exist_ok=True)
os.makedirs(gt_dir,   exist_ok=True)

clear_paths = sorted([
    os.path.join(args.clear_dir, f)
    for f in os.listdir(args.clear_dir)
    if os.path.splitext(f)[1].lower() in IMG_EXTS
])
print(f'Found {len(clear_paths)} clear images → generating '
      f'{len(clear_paths) * args.n_per_image} hazy-clean pairs')


def make_depth_map(img_rgb, sigma_frac=0.05):
    """Outdoor depth proxy: sky/bright = far (1.0), ground/dark = close (0.0)."""
    h, w = img_rgb.shape[:2]

    # vertical gradient: top=far(1.0), bottom=close(0.0)
    vert = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, np.newaxis]
    vert = np.repeat(vert, w, axis=1)

    # luminance proxy
    lum = img_rgb.astype(np.float32).mean(axis=2) / 255.0
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    depth = 0.6 * vert + 0.4 * lum

    # smooth to avoid pixel-level artifacts
    sigma = min(h, w) * sigma_frac
    depth = gaussian_filter(depth, sigma=sigma)

    # renormalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth.astype(np.float32)


def synthesize_haze(img_rgb, A, beta):
    """Apply atmospheric scattering; input/output are uint8 RGB numpy arrays."""
    J     = img_rgb.astype(np.float32) / 255.0
    depth = make_depth_map(img_rgb)
    # depth is normalized [0,1] but SOTS beta values assume metric depth
    # (metres). Multiply by depth_scale to restore realistic magnitude so
    # that far regions (depth≈1) get heavy haze at outdoor beta values.
    t     = np.exp(-beta * depth * args.depth_scale)[:, :, np.newaxis]
    I     = J * t + A * (1.0 - t)
    return np.clip(I * 255.0, 0, 255).astype(np.uint8)


skipped = 0
for clear_path in tqdm(clear_paths, desc='Synthesising'):
    try:
        img_pil = Image.open(clear_path).convert('RGB')
    except Exception:
        skipped += 1
        continue

    img = np.array(img_pil)
    stem = os.path.splitext(os.path.basename(clear_path))[0]

    # save GT once
    gt_out = os.path.join(gt_dir, stem + '.jpg')
    if not os.path.exists(gt_out):
        img_pil.save(gt_out, quality=95)

    # sample n_per_image unique (A, beta) pairs
    pairs = set()
    while len(pairs) < args.n_per_image:
        A    = round(random.uniform(*args.A_range),    2)
        beta = round(random.uniform(*args.beta_range), 3)
        pairs.add((A, beta))

    for A, beta in pairs:
        hazy_name = f'{stem}_{A:.2f}_{beta:.3f}.jpg'
        hazy_out  = os.path.join(hazy_dir, hazy_name)
        if os.path.exists(hazy_out):
            continue
        hazy_img = synthesize_haze(img, A, beta)
        Image.fromarray(hazy_img).save(hazy_out, quality=95)

print(f'\nDone.  Hazy: {len(os.listdir(hazy_dir))}  '
      f'GT: {len(os.listdir(gt_dir))}  '
      f'Skipped: {skipped}')
print(f'\nNext step — update --dehaze_dir in your training scripts to:')
print(f'  {hazy_dir}')
