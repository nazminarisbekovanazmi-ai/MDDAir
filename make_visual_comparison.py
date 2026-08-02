"""
Visual comparison figure generator for MDDAir paper.

Produces a grid in the style of Figure 4 (ICCV/CVPR paper style):
  rows    = degradation types  (Denoising, Deraining, Dehazing, …)
  columns = methods            (Input, PromptIR, …, Ours, GT)

Each cell shows:
  • full image
  • red rectangle crop highlight
  • zoomed inset at the bottom-right corner

Usage
-----
  python make_visual_comparison.py --setting 3d --output comparison_3d.pdf

Edit the CONFIG section below to point at your output folders and pick
representative images / crop boxes.
"""

import os
import argparse
import glob as _glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths and settings
# ═══════════════════════════════════════════════════════════════════════════════

# Root folder where YOUR test script saved its outputs
# (sub-folders: denoise_15, denoise_25, denoise_50, derain, dehaze, deblur, lowlight)
OURS_OUTPUT_ROOT = r"C:\Users\Nazmi\Desktop\실험\MDDAir\output"

# Other methods: (display_name, output_root)
# Each root must have the same sub-folder structure as OURS_OUTPUT_ROOT.
# Files must end with "_restored.png" or just "<stem>.png".
# Leave empty list [] if you only want Input | Ours | GT for now.
OTHER_METHODS_3D = [
    # ("PromptIR",   r"C:\path\to\promptir_output"),
    # ("InstructIR", r"C:\path\to\instructir_output"),
    # ("DFPIR",      r"C:\path\to\dfpir_output"),
]

OTHER_METHODS_5D = [
    # ("InstructIR", r"C:\path\to\instructir_output"),
    # ("IDR",        r"C:\path\to\idr_output"),
    # ("DFPIR",      r"C:\path\to\dfpir_output"),
]

# Degradation rows per setting
ROWS_3D = [
    ("Denoising",  "denoise_25"),
    ("Deraining",  "derain"),
    ("Dehazing",   "dehaze"),
]

ROWS_5D = [
    ("Denoising",  "denoise_25"),
    ("Deraining",  "derain"),
    ("Dehazing",   "dehaze"),
    ("Deblurring", "deblur"),
    ("Low-light",  "lowlight"),
]

# Which image stem to pick per degradation sub-folder.
# None  → picks the first alphabetical file found.
# str   → exact stem (no suffix), e.g. "im_00001"
SELECTED_IMAGE = {
    "denoise_25": None,
    "denoise_15": None,
    "denoise_50": None,
    "derain":     None,
    "dehaze":     None,
    "deblur":     None,
    "lowlight":   None,
}

# Crop box per degradation: (left, top, right, bottom) in PIXEL coords of original image.
# None → auto-picks a region with high local variance (interesting area).
CROP_BOXES = {
    "denoise_25": None,
    "denoise_15": None,
    "denoise_50": None,
    "derain":     None,
    "dehaze":     None,
    "deblur":     None,
    "lowlight":   None,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE STYLE
# ═══════════════════════════════════════════════════════════════════════════════

CELL_SIZE_INCH = 2.0        # each image cell (width = height)
ROW_LABEL_WIDTH = 0.55      # left column for degradation label
METHOD_LABEL_HEIGHT = 0.40  # bottom row for method names
INSET_FRACTION = 0.35       # zoom inset size as fraction of cell
ZOOM_FACTOR = 2.0           # magnification applied inside the inset
CROP_COLOR = "#ff3333"
CROP_LW = 1.5
FONT_SIZE_METHOD = 8
FONT_SIZE_ROW = 9
DPI = 300

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_png(path):
    """Return HxWxC float32 [0,1] array, or None if file missing."""
    if path and os.path.isfile(path):
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    return None


def _find_stem(folder, stem):
    """Return the first PNG in folder whose stem matches (ignoring suffix like _psnr)."""
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".png"):
            continue
        name = f
        # strip known suffixes so "im001_28.45.png" and "im001_restored.png" both match "im001"
        for suf in ("_restored", "_input", "_gt"):
            name = name.replace(suf, "")
        # name is now like "im001_28.45.png" or "im001.png"
        if name.split("_")[0] == stem.split("_")[0] or name.startswith(stem):
            return f.replace(".png", "").split("_restored")[0].split("_input")[0].split("_gt")[0]
    return None


def _pick_stem(folder):
    """Return the stem of the first valid image (not _input / _gt / PSNR-named heuristic)."""
    candidates = []
    for f in sorted(os.listdir(folder)):
        if f.endswith("_restored.png"):
            candidates.append(f.replace("_restored.png", ""))
        elif f.endswith(".png") and "_input" not in f and "_gt" not in f:
            # PSNR-named: stem_28.45.png → take anything
            candidates.append(os.path.splitext(f)[0])
    return candidates[0] if candidates else None


def _auto_crop(img_hw3, frac=0.25):
    """Return (l,t,r,b) of a high-variance region sized frac × frac of the image."""
    H, W = img_hw3.shape[:2]
    ch, cw = int(H * frac), int(W * frac)
    gray = img_hw3.mean(axis=2)
    best_var, best_pos = -1, (0, 0)
    step = max(8, min(H, W) // 16)
    for y in range(0, H - ch, step):
        for x in range(0, W - cw, step):
            v = gray[y:y+ch, x:x+cw].var()
            if v > best_var:
                best_var, best_pos = v, (x, y)
    x0, y0 = best_pos
    return (x0, y0, x0 + cw, y0 + ch)


def _load_cell(output_root, sub_folder, stem, kind):
    """
    kind: "restored" | "input" | "gt"
    Returns np.array or None.
    """
    folder = os.path.join(output_root, sub_folder)
    if not os.path.isdir(folder):
        return None

    # First try clean-named versions saved by updated test scripts
    clean = os.path.join(folder, f"{stem}_{kind}.png")
    if kind == "restored":
        clean = os.path.join(folder, f"{stem}_restored.png")
    if os.path.isfile(clean):
        return _load_png(clean)

    # Fallback: for "restored", find any file that starts with stem and isn't input/gt
    if kind == "restored":
        for f in sorted(os.listdir(folder)):
            if f.startswith(stem) and "_input" not in f and "_gt" not in f and f.endswith(".png"):
                return _load_png(os.path.join(folder, f))
    return None


def _draw_cell(ax, img, crop_box, inset_frac, zoom_factor, method_name=None):
    """Draw one comparison cell: image + red crop box + zoom inset."""
    H, W = img.shape[:2]
    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if crop_box is None:
        crop_box = _auto_crop(img)

    l, t, r, b = crop_box
    # clamp to image bounds
    l, r = max(0, l), min(W, r)
    t, b = max(0, t), min(H, b)
    cw, ch = r - l, b - t

    rect = mpatches.Rectangle(
        (l, t), cw, ch,
        linewidth=CROP_LW, edgecolor=CROP_COLOR, facecolor="none",
        transform=ax.transData, zorder=5
    )
    ax.add_patch(rect)

    # Zoom inset — placed at bottom-right corner
    inset_ax = ax.inset_axes(
        [1.0 - inset_frac - 0.01, 0.01, inset_frac, inset_frac],
        transform=ax.transAxes
    )
    crop_img = img[t:b, l:r]
    inset_ax.imshow(crop_img, aspect="auto", interpolation="nearest")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(CROP_COLOR)
        spine.set_linewidth(CROP_LW)
        spine.set_visible(True)

    if method_name:
        ax.text(
            0.5, 0.02, method_name,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=FONT_SIZE_METHOD, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def build_figure(rows, methods_def, output_path):
    """
    rows        : list of (row_label, sub_folder)
    methods_def : list of (col_label, output_root, kind)
                  kind = "restored" | "input" | "gt"
    """
    n_rows = len(rows)
    n_cols = len(methods_def)

    fig_w = ROW_LABEL_WIDTH + n_cols * CELL_SIZE_INCH
    fig_h = n_rows * CELL_SIZE_INCH + METHOD_LABEL_HEIGHT

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        n_rows, n_cols,
        left=ROW_LABEL_WIDTH / fig_w,
        right=1.0,
        top=1.0 - METHOD_LABEL_HEIGHT / fig_h,
        bottom=METHOD_LABEL_HEIGHT / fig_h,
        hspace=0.04,
        wspace=0.04,
    )

    for ri, (row_label, sub_folder) in enumerate(rows):
        # Determine which stem to use
        selected = SELECTED_IMAGE.get(sub_folder)
        ref_folder = os.path.join(OURS_OUTPUT_ROOT, sub_folder)
        if selected is None and os.path.isdir(ref_folder):
            selected = _pick_stem(ref_folder)
        if selected is None:
            print(f"  [WARN] No images found in {ref_folder} — skipping row '{row_label}'")
            continue

        # Determine crop box from reference (Ours restored image)
        crop = CROP_BOXES.get(sub_folder)
        ref_img = _load_cell(OURS_OUTPUT_ROOT, sub_folder, selected, "restored")
        if ref_img is None:
            ref_img = _load_cell(OURS_OUTPUT_ROOT, sub_folder, selected, "input")
        if crop is None and ref_img is not None:
            crop = _auto_crop(ref_img)

        for ci, (col_label, out_root, kind) in enumerate(methods_def):
            ax = fig.add_subplot(gs[ri, ci])

            img = _load_cell(out_root, sub_folder, selected, kind)
            if img is None:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#888888")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # Show method name only on last row (cleaner than every cell)
            show_name = (ri == n_rows - 1)
            _draw_cell(ax, img, crop, INSET_FRACTION, ZOOM_FACTOR,
                       method_name=None)

        # Row label (rotated, left of first column)
        fig.text(
            (ROW_LABEL_WIDTH / fig_w) * 0.5,
            1.0 - METHOD_LABEL_HEIGHT / fig_h - (ri + 0.5) * CELL_SIZE_INCH / fig_h,
            row_label,
            ha="center", va="center",
            fontsize=FONT_SIZE_ROW, fontweight="bold", rotation=90,
            transform=fig.transFigure
        )

    # Column method names at the bottom
    col_x_step = (1.0 - ROW_LABEL_WIDTH / fig_w) / n_cols
    col_x_start = ROW_LABEL_WIDTH / fig_w + col_x_step * 0.5
    y_label = METHOD_LABEL_HEIGHT / fig_h * 0.45
    for ci, (col_label, _, _) in enumerate(methods_def):
        fig.text(
            col_x_start + ci * col_x_step,
            y_label,
            col_label,
            ha="center", va="center",
            fontsize=FONT_SIZE_METHOD + 1, fontweight="bold",
            transform=fig.transFigure
        )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build visual comparison figure")
    parser.add_argument("--setting", choices=["3d", "5d"], default="3d")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (PNG or PDF). Default: comparison_<setting>.pdf")
    args = parser.parse_args()

    if args.setting == "3d":
        rows = ROWS_3D
        other = OTHER_METHODS_3D
    else:
        rows = ROWS_5D
        other = OTHER_METHODS_5D

    out_path = args.output or f"comparison_{args.setting}.pdf"

    # Build method columns: Input | <other methods> | Ours | GT
    methods = [("Input", OURS_OUTPUT_ROOT, "input")]
    for name, root in other:
        methods.append((name, root, "restored"))
    methods.append(("Ours", OURS_OUTPUT_ROOT, "restored"))
    methods.append(("GT",   OURS_OUTPUT_ROOT, "gt"))

    print(f"Setting : {args.setting.upper()}")
    print(f"Rows    : {[r[0] for r in rows]}")
    print(f"Columns : {[m[0] for m in methods]}")
    print()

    build_figure(rows, methods, out_path)


if __name__ == "__main__":
    main()
