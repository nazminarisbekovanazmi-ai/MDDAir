import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from net.model import ChannelShuffle_skip_textguaid
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
import clip


parser = argparse.ArgumentParser(description='Test 3D')
parser.add_argument('--gpu', type=str, default="0", help='GPUs')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--pretrained_1', type=str, default='./', help='path to checkpoint')
parser.add_argument('--denoise_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\BSD68")
parser.add_argument('--denoise_path2', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Urban100\image_SRF_4")
parser.add_argument('--derain_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Rain\Rain100L_test\rain_test")
parser.add_argument('--dehaze_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\hazy_test\hazy_outdoor")
parser.add_argument('--output_path', type=str, default="output/")
parser.add_argument('--ablation', type=str, default='full',
                    choices=['full', 'no_deg_estimator', 'no_spatial_attn', 'no_film'],
                    help='must match the --ablation variant the checkpoint was trained with.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = device == "cuda"
print("Using device:", device)

clip_model, _ = clip.load("ViT-B/32", device=device)
for param in clip_model.parameters():
    param.requires_grad = False

inputext = [
    "Gaussian noise with a standard deviation of 15",
    "Gaussian noise with a standard deviation of 25",
    "Gaussian noise with a standard deviation of 50",
    "Rain degradation with rain lines",
    "Hazy degradation with normal haze",
    "Blur degradation with motion blur",
    "Lowlight degradation",
]

with torch.no_grad():
    _text_tokens_all = clip.tokenize(inputext).to(device)
    text_features_all = clip_model.encode_text(_text_tokens_all).to(dtype=torch.float32)
    text_features_all = F.normalize(text_features_all, dim=-1)


def _run_one(model, loader, true_idx, output_dir):
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
            save_image_tensor(restored, os.path.join(output_dir, img_name + f"_{temp_psnr:.2f}.png"))
            # clean-named copies for comparison figure (no PSNR in name → same stem across methods)
            save_image_tensor(restored,      os.path.join(output_dir, img_name + "_restored.png"))
            save_image_tensor(degrad_patch,  os.path.join(output_dir, img_name + "_input.png"))
            save_image_tensor(clean_patch,   os.path.join(output_dir, img_name + "_gt.png"))

    return psnr_m.avg, ssim_m.avg, correct, total, acc_probs / total


def test_all(model):
    """Structured test: per-type PSNR/SSIM and degradation-type detection accuracy."""
    TYPE_NAMES = ["N-15", "N-25", "N-50", "Rain", "Haze", "Blur", "Low"]
    model.eval()

    noise_set  = DenoiseTestDataset(args)
    derain_set = DerainDehazeDataset(args, addnoise=False, sigma=15)

    def make_loader(dataset):
        return DataLoader(dataset, batch_size=1, pin_memory=pin_memory, shuffle=False, num_workers=0)

    rows = []

    for sigma, true_idx in [(15, 0), (25, 1), (50, 2)]:
        noise_set.set_sigma(sigma)
        stats = _run_one(model, make_loader(noise_set), true_idx,
                         os.path.join(args.output_path, f"denoise_{sigma}"))
        rows.append((f"Noise s={sigma}", true_idx, *stats))

    derain_set.set_dataset("derain")
    rows.append(("Rain", 3, *_run_one(model, make_loader(derain_set), 3,
                                       os.path.join(args.output_path, "derain"))))

    derain_set.set_dataset("dehaze")
    rows.append(("Haze", 4, *_run_one(model, make_loader(derain_set), 4,
                                       os.path.join(args.output_path, "dehaze"))))

    prob_header = "  ".join(f"{n:>5}" for n in TYPE_NAMES)
    header = f"{'Task':<14}  {'PSNR':>6}  {'SSIM':>6}  {'Det':>9}  {prob_header}  Type?"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    all_psnr, all_ssim = [], []
    for name, true_idx, psnr, ssim, correct, total, avg_p in rows:
        pred_idx = avg_p.argmax().item()
        flag = "OK" if pred_idx == true_idx else f"X -> {TYPE_NAMES[pred_idx]}"
        prob_str = "  ".join(f"{v:>5.2f}" for v in avg_p.tolist())
        print(f"{name:<14}  {psnr:>6.2f}  {ssim:>6.4f}  {correct:>4}/{total:<4}  {prob_str}  {flag}")
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    print(sep)
    avg_psnr = sum(all_psnr) / len(all_psnr)   # (N15+N25+N50+Rain+Haze)/5 — matches paper Table 1
    avg_ssim = sum(all_ssim) / len(all_ssim)
    print(f"{'Average (3-task)':<14}  {avg_psnr:>6.2f}  {avg_ssim:>6.4f}")
    print(sep + "\n")

    # ── save results table figure (academic style) ───────────────────────────
    os.makedirs(args.output_path, exist_ok=True)
    task_labels = [r[0] for r in rows] + ["Average"]
    psnr_vals   = [r[2] for r in rows] + [avg_psnr]
    ssim_vals   = [r[3] for r in rows] + [avg_ssim]
    det_vals    = [f"{r[4]}/{r[5]}" for r in rows] + ["—"]

    col_labels = ["Task", "PSNR ↑", "SSIM ↑", "Det. Acc"]
    table_data = [[t, f"{p:.2f}", f"{s:.4f}", d]
                  for t, p, s, d in zip(task_labels, psnr_vals, ssim_vals, det_vals)]

    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.15, 1.9)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#f0f0f0')
        tbl[0, j].set_text_props(fontweight='bold', color='black')
        tbl[0, j].set_edgecolor('#aaaaaa')

    for i in range(1, n_rows + 1):
        is_avg = (i == n_rows)
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#f7f7f7' if is_avg else 'white')
            tbl[i, j].set_edgecolor('#dddddd')
            if is_avg:
                tbl[i, j].set_text_props(fontweight='bold')

    plt.title("MDDAir — 3-task Results", fontsize=11, fontweight='bold',
              pad=10, color='#222222')
    fig_path = os.path.join(args.output_path, "results_table_3task.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close()
    print(f"Results figure saved to {fig_path}")

    return avg_psnr, avg_ssim


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    np.random.seed(0)
    torch.manual_seed(0)

    ablation_kwargs = {
        'full':             {},
        'no_deg_estimator': {'use_deg_estimator': False},
        'no_spatial_attn':  {'use_spatial_attn': False},
        'no_film':          {'use_film': False},
    }[args.ablation]
    print(f"==> ablation variant: {args.ablation}")
    model = ChannelShuffle_skip_textguaid(text_features_all=text_features_all, **ablation_kwargs)
    model.to(device)

    if args.pretrained_1 and os.path.isfile(args.pretrained_1):
        print("=> loading model '{}'".format(args.pretrained_1))
        model_pretrained = torch.load(args.pretrained_1, map_location=device)
        model_state = model.state_dict()
        compatible_dict = {
            k: v for k, v in model_pretrained['state_dict'].items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = set(model_pretrained['state_dict']) - set(compatible_dict)
        if skipped:
            print(f"    skipping {len(skipped)} incompatible key(s) (architecture changed): {sorted(skipped)}")
        model.load_state_dict(compatible_dict, strict=False)
    else:
        print("=> no model found at '{}'".format(args.pretrained_1))

    test_all(model)
