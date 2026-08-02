"""Build a qualitative comparison figure in the classic restoration-paper style:
Input (full image, red box marking a region) -> each method's zoomed crop of that
region -> Ours (highlighted) -> GT."""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.metrics import structural_similarity as _ssim

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # serif-matching math glyphs (e.g. sigma)

BASE_DIR = Path(
    r"C:\Users\Nazmi\Desktop\MDDAir\Qualitive Results\Qualitive comparison illustration\5_deg_comparison"
)

# Row label override (plain degradation name used when not listed here)
ROW_LABELS = {
    "Noise": r"Noise ($\sigma$=25)",
}

# (key, filename regex, display label) — checked in order; a file matched by an
# earlier spec is removed from the pool so later specs can't also claim it (needed
# because e.g. Blur's "802_Deg_GT.png" contains both "deg" and "gt").
METHOD_SPECS = [
    ("deg_img", r"deg", "Input"),
    ("restormer", r"restormer", "Restormer"),
    ("promptir", r"promptir", "PromptIR"),
    ("instructir", r"instructir", "InstructIR"),
    ("dfpir", r"dfpir", "DFPIR"),
    ("mddair", r"mddair", "MDDAIR (Ours)"),
    ("gt", r"gt", "GT"),
]

# crop box (left, top, right, bottom) marking the zoomed region, per degradation folder
CROP_BOXES = {
    "Rain": (10, 140, 150, 260),
    "Haze": (180, 260, 340, 420),
    "Noise": (100, 800, 320, 1000),
    "Blur": (1030, 180, 1280, 280),
    "Low-light": (300, 150, 500, 300),
}


def _extract_psnr(stem: str):
    decimals = re.findall(r"\d+\.\d+", stem)
    return float(decimals[-1]) if decimals else None


def _compute_ssim(path: Path, gt_path: Path):
    img = Image.open(path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")
    if img.size != gt.size:
        img = img.resize(gt.size, Image.BICUBIC)
    a = np.asarray(img, dtype=np.float64)
    b = np.asarray(gt, dtype=np.float64)
    return float(_ssim(a, b, channel_axis=2, data_range=255))


# Only these methods ever carry a real PSNR in their filename across all folders;
# other methods' filenames can contain unrelated decimals (e.g. Haze's synthesis
# params "0111_0.8_0.2(InstructIR).jpg"), which must NOT be mistaken for a score.
PSNR_KEYS = {"dfpir", "mddair", "promptir", "instructir"}

# Folders where Deg_IMG.png / *_GT.png are mislabeled relative to their actual content
# (verified visually: Noise's "Deg_IMG.png" is clean and its "*_GT.png" is noisy) —
# swap Input <-> GT after matching so the figure reflects what the files actually show.
SWAP_INPUT_GT = {"Noise"}


def find_method_files(folder: Path):
    """Returns {key: (path, label, psnr_or_None, ssim_or_None)}."""
    remaining = [f for f in folder.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    resolved = {}
    for key, pattern, label in METHOD_SPECS:
        match = next((f for f in remaining if re.search(pattern, f.stem, re.IGNORECASE)), None)
        if match is None:
            continue
        remaining.remove(match)
        psnr = _extract_psnr(match.stem) if key in PSNR_KEYS else None
        resolved[key] = [match, label, psnr, None]

    if folder.name in SWAP_INPUT_GT and "deg_img" in resolved and "gt" in resolved:
        resolved["deg_img"], resolved["gt"] = (
            [resolved["gt"][0], resolved["deg_img"][1], resolved["gt"][2], resolved["gt"][3]],
            [resolved["deg_img"][0], resolved["gt"][1], resolved["deg_img"][2], resolved["deg_img"][3]],
        )

    if "gt" in resolved:
        gt_path = resolved["gt"][0]
        for key in PSNR_KEYS:
            if key in resolved:
                resolved[key][3] = _compute_ssim(resolved[key][0], gt_path)

    return {k: tuple(v) for k, v in resolved.items()}


def build_figure(folder: Path, crop_box, out_path: Path):
    methods = find_method_files(folder)
    order = [spec[0] for spec in METHOD_SPECS if spec[0] in methods]
    missing = [label for key, _, label in METHOD_SPECS if key not in methods]
    if missing:
        print(f"  ({folder.name}: no file found for {', '.join(missing)} — skipping those columns)")
    n = len(order)

    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.5), gridspec_kw={"wspace": 0.02})

    for ax, key in zip(axes, order):
        path, label, psnr, ssim = methods[key]
        img = Image.open(path).convert("RGB")

        if key == "deg_img":
            ax.imshow(img)
            l, t, r, b = crop_box
            ax.add_patch(Rectangle((l, t), r - l, b - t, edgecolor="red", facecolor="none", linewidth=2))
        else:
            ax.imshow(img.crop(crop_box))

        ax.set_xticks([])
        ax.set_yticks([])

        is_ours = key == "mddair"
        for spine in ax.spines.values():
            spine.set_visible(is_ours)
            if is_ours:
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

        caption = label if psnr is None else f"{label}\n{psnr:.2f} dB / {ssim:.4f}"
        ax.set_title(
            caption,
            fontsize=20,
            fontweight="bold" if is_ours else "normal",
            color="red" if is_ours else "black",
        )

    fig.subplots_adjust(wspace=0.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def build_combined_figure(degradations, out_path: Path, title: str = None):
    """One grid: rows = degradation types, columns = union of methods seen across
    them. Cells for a method missing in a given row's folder are left blank."""
    per_row = {d: find_method_files(BASE_DIR / d) for d in degradations}
    order = [key for key, _, _ in METHOD_SPECS if any(key in m for m in per_row.values())]
    n_cols = len(order)
    n_rows = len(degradations)

    # Size each row's height to its own crop's aspect ratio (h/w), not a fixed value —
    # otherwise a short/wide crop (e.g. Blur) gets vertically centered with dead white
    # space in a cell sized for a squarer crop (e.g. LOL).
    col_width = 5.5 
    height_ratios = [
        (CROP_BOXES[d][3] - CROP_BOXES[d][1]) / (CROP_BOXES[d][2] - CROP_BOXES[d][0])
        for d in degradations
    ]
    fig_height = col_width * sum(height_ratios)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, fig_height),
                              gridspec_kw={"wspace": 0.02, "hspace": 0.22, "height_ratios": height_ratios})
    if n_rows == 1:
        axes = axes[None, :]

    label_by_key = {key: label for key, _, label in METHOD_SPECS}

    for i, degradation in enumerate(degradations):
        methods = per_row[degradation]
        crop_box = CROP_BOXES[degradation]

        for j, key in enumerate(order):
            ax = axes[i, j]

            if key not in methods:
                ax.axis("off")
                continue

            path, _, psnr, ssim = methods[key]
            img = Image.open(path).convert("RGB")

            if key == "deg_img":
                ax.imshow(img)
                l, t, r, b = crop_box
                ax.add_patch(Rectangle((l, t), r - l, b - t, edgecolor="red", facecolor="none", linewidth=2))
            else:
                ax.imshow(img.crop(crop_box))

            ax.set_xticks([])
            ax.set_yticks([])

            is_ours = key == "mddair"
            for spine in ax.spines.values():
                spine.set_visible(is_ours)
                if is_ours:
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)

            if psnr is not None:
                ax.set_xlabel(f"{psnr:.2f} dB / {ssim:.4f}", fontsize=20,
                              fontweight="bold" if is_ours else "normal",
                              color="red" if is_ours else "black")

            if i == 0:
                ax.set_title(label_by_key[key], fontsize=20,
                             fontweight="bold" if is_ours else "normal",
                             color="red" if is_ours else "black")

            if j == 0:
                ax.set_ylabel(ROW_LABELS.get(degradation, degradation), fontsize=11, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=25, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.02, hspace=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved combined figure to {out_path}")


if __name__ == "__main__":
    degradations = ["Rain", "Haze", "Noise", "Blur", "Low-light"]
    for degradation in degradations:
        folder = BASE_DIR / degradation
        out_path = folder / f"{degradation.lower()}_comparison.pdf"
        build_figure(folder, CROP_BOXES[degradation], out_path)

    build_combined_figure(
        ["Noise", "Rain", "Haze"],
        BASE_DIR / "comparison_noise_rain_haze.pdf",
    )
    build_combined_figure(
        ["Blur", "Low-light"],
        BASE_DIR / "comparison_blur_lowlight.pdf",
    )
