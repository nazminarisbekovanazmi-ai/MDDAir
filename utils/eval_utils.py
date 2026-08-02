"""Shared evaluation logic used by both train_5D_MDDAir.py and test_5D_MDDAir.py."""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_utils import (
    DenoiseTestDataset, DerainDehazeDataset, DeblurTestDataset, LOLTestDataset,
)
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor


def _run_one(model, loader, true_idx, output_dir, device):
    """One dataset pass: PSNR/SSIM + type-detection accuracy in a single loop.

    No type label is passed to the model — type_probs is read back from
    DegradationEstimator and compared against true_idx for accuracy.
    """
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    correct, total = 0, 0
    acc_probs = torch.zeros(7)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for (name_list, degrad_patch, clean_patch) in tqdm(loader, leave=False):
            degrad_patch = degrad_patch.to(device)
            clean_patch  = clean_patch.to(device)
            restored, _, type_logits = model(degrad_patch)

            type_probs_cpu = torch.sigmoid(type_logits).cpu().squeeze(0)
            acc_probs += type_probs_cpu
            if type_probs_cpu.argmax().item() == true_idx:
                correct += 1
            total += 1

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr_m.update(temp_psnr, N)
            ssim_m.update(temp_ssim, N)

            img_name = name_list[0][0]
            save_image_tensor(
                restored,
                os.path.join(output_dir, img_name + f"_{temp_psnr:.2f}.png"),
            )

    return psnr_m.avg, ssim_m.avg, correct, total, acc_probs / total


def eval_all(model, args, device, pin_memory=False):
    """Structured evaluation across all 7 degradation types.

    Returns a dict with per-task PSNR/SSIM values and 5-task averages,
    plus the raw rows list for table printing.

    Used by training (checkpoint naming) and testing (reporting).
    """
    model.eval()

    noise_set  = DenoiseTestDataset(args)
    derain_set = DerainDehazeDataset(args, addnoise=False, sigma=15)
    deblur_set = DeblurTestDataset(args, addnoise=False, sigma=15)
    lol_set    = LOLTestDataset(args, addnoise=False, sigma=15)

    def make_loader(dataset):
        return DataLoader(dataset, batch_size=1, pin_memory=pin_memory,
                          shuffle=False, num_workers=0)

    rows = []

    for sigma, true_idx in [(15, 0), (25, 1), (50, 2)]:
        noise_set.set_sigma(sigma)
        stats = _run_one(model, make_loader(noise_set), true_idx,
                         os.path.join(args.output_path, f"denoise_{sigma}"), device)
        rows.append((f"Noise σ={sigma}", true_idx, *stats))

    derain_set.set_dataset("derain")
    rows.append(("Rain", 3, *_run_one(model, make_loader(derain_set), 3,
                                       os.path.join(args.output_path, "derain"), device)))

    derain_set.set_dataset("dehaze")
    rows.append(("Haze", 4, *_run_one(model, make_loader(derain_set), 4,
                                       os.path.join(args.output_path, "dehaze"), device)))

    rows.append(("Blur", 5, *_run_one(model, make_loader(deblur_set), 5,
                                       os.path.join(args.output_path, "deblur"), device)))

    rows.append(("Lowlight", 6, *_run_one(model, make_loader(lol_set), 6,
                                           os.path.join(args.output_path, "lowlight"), device)))

    psnr_noise = sum(r[2] for r in rows[:3]) / 3
    ssim_noise = sum(r[3] for r in rows[:3]) / 3
    psnr_rain,  ssim_rain  = rows[3][2], rows[3][3]
    psnr_haze,  ssim_haze  = rows[4][2], rows[4][3]
    psnr_blur,  ssim_blur  = rows[5][2], rows[5][3]
    psnr_lol,   ssim_lol   = rows[6][2], rows[6][3]
    psnr_avr = (psnr_noise + psnr_rain + psnr_haze + psnr_blur + psnr_lol) / 5
    ssim_avr = (ssim_noise + ssim_rain + ssim_haze + ssim_blur + ssim_lol) / 5

    return {
        'rows':       rows,
        'psnr_noise': psnr_noise, 'ssim_noise': ssim_noise,
        'psnr_rain':  psnr_rain,  'ssim_rain':  ssim_rain,
        'psnr_haze':  psnr_haze,  'ssim_haze':  ssim_haze,
        'psnr_blur':  psnr_blur,  'ssim_blur':  ssim_blur,
        'psnr_lol':   psnr_lol,   'ssim_lol':   ssim_lol,
        'psnr_avr':   psnr_avr,   'ssim_avr':   ssim_avr,
    }


def eval_all_3D(model, args, device, pin_memory=False):
    """3D variant: evaluates noise (σ15/25/50), rain, haze — 3 task categories.

    Returns dict with per-task PSNR/SSIM and 3-task average.
    Used by train_3D_MDDAir.py and test_3D_MDDAir.py.
    """
    from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset

    model.eval()

    noise_set  = DenoiseTestDataset(args)
    derain_set = DerainDehazeDataset(args, addnoise=False, sigma=15)

    def make_loader(dataset):
        return DataLoader(dataset, batch_size=1, pin_memory=pin_memory,
                          shuffle=False, num_workers=0)

    rows = []

    for sigma, true_idx in [(15, 0), (25, 1), (50, 2)]:
        noise_set.set_sigma(sigma)
        stats = _run_one(model, make_loader(noise_set), true_idx,
                         os.path.join(args.output_path, f"denoise_{sigma}"), device)
        rows.append((f"Noise σ={sigma}", true_idx, *stats))

    derain_set.set_dataset("derain")
    rows.append(("Rain", 3, *_run_one(model, make_loader(derain_set), 3,
                                       os.path.join(args.output_path, "derain"), device)))

    derain_set.set_dataset("dehaze")
    rows.append(("Haze", 4, *_run_one(model, make_loader(derain_set), 4,
                                       os.path.join(args.output_path, "dehaze"), device)))

    psnr_n15, ssim_n15 = rows[0][2], rows[0][3]
    psnr_n25, ssim_n25 = rows[1][2], rows[1][3]
    psnr_n50, ssim_n50 = rows[2][2], rows[2][3]
    psnr_noise = (psnr_n15 + psnr_n25 + psnr_n50) / 3
    ssim_noise = (ssim_n15 + ssim_n25 + ssim_n50) / 3
    psnr_rain, ssim_rain = rows[3][2], rows[3][3]
    psnr_haze, ssim_haze = rows[4][2], rows[4][3]
    psnr_avr = (psnr_n15 + psnr_n25 + psnr_n50 + psnr_rain + psnr_haze) / 5
    ssim_avr = (ssim_n15 + ssim_n25 + ssim_n50 + ssim_rain + ssim_haze) / 5

    return {
        'rows':       rows,
        'psnr_n15':   psnr_n15,   'ssim_n15':   ssim_n15,
        'psnr_n25':   psnr_n25,   'ssim_n25':   ssim_n25,
        'psnr_n50':   psnr_n50,   'ssim_n50':   ssim_n50,
        'psnr_noise': psnr_noise, 'ssim_noise': ssim_noise,
        'psnr_rain':  psnr_rain,  'ssim_rain':  ssim_rain,
        'psnr_haze':  psnr_haze,  'ssim_haze':  ssim_haze,
        'psnr_avr':   psnr_avr,   'ssim_avr':   ssim_avr,
    }
