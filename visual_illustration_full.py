"""Qualitative illustrations drawn from the FULL test-set output folders (not the
small curated 5_deg_comparison examples), reusing qualitive_comparison.py's exact
visual style: Input with a red crop-box, each method's zoomed crop of that region,
MDDAIR (Ours) highlighted in red, PSNR/SSIM printed below each scored method.

Produces:
  - 4 illustrations covering Noise(Urban100,sigma=25)/Rain/Haze
    columns: Deg_IMG, PromptIR, InstructIR, DFPIR, MDDAIR (Ours), GT
  - 4 illustrations covering Blur/Low-light
    columns: Deg_IMG, InstructIR, DFPIR, MDDAIR (Ours), GT   (no PromptIR: it has
    no deblur/lowlight results)

Each illustration draws a fresh random image per row; the crop box is picked
automatically (highest local-variance patch) since these images change every run.
"""
import random
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

QUAL_BASE = Path(r"C:\Users\Nazmi\Desktop\MDDAir\Qualitive Results")
DEG_BASE = QUAL_BASE / "Degradation_datasets"
OUT_DIR = QUAL_BASE / "Full Illustrations"

SEED = 7
IMG_EXTS = (".png", ".jpg", ".jpeg")
NOISE_SIGMA = 25


def _extract_psnr(stem: str):
    m = re.search(r"psnr_?(\d+\.\d+)", stem, re.IGNORECASE)
    if m:
        return float(m.group(1))
    decimals = re.findall(r"\d+\.\d+", stem)
    return float(decimals[-1]) if decimals else None


def _list_ids(folder: Path, id_pattern):
    ids = set()
    for f in folder.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        m = id_pattern.search(f.stem)
        if m:
            ids.add(m.group(1))
    return ids


def _best_file_for_id(folder: Path, id_pattern, target_id: str):
    scored, unscored = [], []
    for f in folder.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        m = id_pattern.search(f.stem)
        if not m or m.group(1) != target_id:
            continue
        psnr = _extract_psnr(f.stem)
        (scored if psnr is not None else unscored).append((f, psnr))
    if scored:
        return max(scored, key=lambda x: x[1])[0]
    return unscored[0][0] if unscored else None


def _find_ext(folder: Path, stem: str):
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _synth_noisy(gt_path: Path, sigma: float, seed: int):
    clean = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float64)
    rng = np.random.RandomState(seed)
    noise = rng.randn(*clean.shape) * sigma
    noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _compute_ssim(img: Image.Image, gt_img: Image.Image):
    if img.size != gt_img.size:
        img = img.resize(gt_img.size, Image.BICUBIC)
    a = np.asarray(img, dtype=np.float64)
    b = np.asarray(gt_img, dtype=np.float64)
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def _auto_crop(img: Image.Image, frac=0.3):
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    H, W = arr.shape
    ch, cw = int(H * frac), int(W * frac)
    step = max(8, min(H, W) // 20)
    best_var, best = -1, (0, 0)
    for y in range(0, H - ch, step):
        for x in range(0, W - cw, step):
            v = arr[y:y + ch, x:x + cw].var()
            if v > best_var:
                best_var, best = v, (x, y)
    x0, y0 = best
    return (x0, y0, x0 + cw, y0 + ch)


TASKS = {
    "Noise": {
        "folders": {
            "PromptIR": QUAL_BASE / "PromptIR" / "denoise" / "25",
            "InstructIR": QUAL_BASE / "InstructIR" / "Urban100_25",
            "DFPIR": QUAL_BASE / "DFPIR" / "noise(25)",
            "MDDAir": QUAL_BASE / "MDDAir" / "denoise_25",
        },
        "id_pattern": re.compile(r"img_(\d+)_SRF", re.IGNORECASE),
        "gt_path": lambda id_: DEG_BASE / "Urban100" / "image_SRF_4" / f"img_{id_}_SRF_4_HR.png",
        "synthesize_input": True,
    },
    "Rain": {
        "folders": {
            "PromptIR": QUAL_BASE / "PromptIR" / "derain",
            "InstructIR": QUAL_BASE / "InstructIR" / "Rain100L",
            "DFPIR": QUAL_BASE / "DFPIR" / "derain",
            "MDDAir": QUAL_BASE / "MDDAir" / "derain",
        },
        "id_pattern": re.compile(r"(?:rain|norain)-(\d+)"),
        "gt_path": lambda id_: DEG_BASE / "Rain" / "Rain100L_test" / "GT_rain_test" / f"norain-{id_}.png",
        "input_path": lambda id_: DEG_BASE / "Rain" / "Rain100L_test" / "rain_test" / f"rain-{id_}.png",
    },
    "Haze": {
        "folders": {
            "PromptIR": QUAL_BASE / "PromptIR" / "dehaze",
            "InstructIR": QUAL_BASE / "InstructIR" / "SOTS",
            "DFPIR": QUAL_BASE / "DFPIR" / "dehaze",
            "MDDAir": QUAL_BASE / "MDDAir" / "dehaze",
        },
        "id_pattern": re.compile(r"^(\d{4}_[\d.]+_[\d.]+)"),
        "gt_path": lambda id_: _find_ext(DEG_BASE / "hazy_test" / "GT_outdoor", id_),
        "input_path": lambda id_: _find_ext(DEG_BASE / "hazy_test" / "hazy_outdoor", id_),
    },
    "Blur": {
        "folders": {
            "InstructIR": QUAL_BASE / "InstructIR" / "GoPro",
            "DFPIR": QUAL_BASE / "DFPIR" / "deblur",
            "MDDAir": QUAL_BASE / "MDDAir" / "deblur",
        },
        "id_pattern": re.compile(r"^(\d+)[_.]"),
        "gt_path": lambda id_: DEG_BASE / "Gopro" / "test" / "sharp" / f"{id_}.png",
        "input_path": lambda id_: DEG_BASE / "Gopro" / "test" / "blur" / f"{id_}.png",
    },
    "Low-light": {
        "folders": {
            "InstructIR": QUAL_BASE / "InstructIR" / "LOL",
            "DFPIR": QUAL_BASE / "DFPIR" / "lowlight",
            "MDDAir": QUAL_BASE / "MDDAir" / "lowlight",
        },
        "id_pattern": re.compile(r"^(\d+)[_.]"),
        "gt_path": lambda id_: DEG_BASE / "lol_dataset" / "eval15" / "high" / f"{id_}.png",
        "input_path": lambda id_: DEG_BASE / "lol_dataset" / "eval15" / "low" / f"{id_}.png",
    },
}

METHOD_LABELS = {
    "promptir": "PromptIR",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "mddair": "MDDAIR (Ours)",
}
METHOD_KEY_BY_FOLDER_NAME = {"PromptIR": "promptir", "InstructIR": "instructir", "DFPIR": "dfpir", "MDDAir": "mddair"}


def common_ids(task: str):
    cfg = TASKS[task]
    id_sets = [_list_ids(folder, cfg["id_pattern"]) for folder in cfg["folders"].values()]
    return sorted(set.intersection(*id_sets))


def mddair_winning_ids(task: str):
    """IDs (from common_ids) where MDDAir's embedded PSNR is strictly the highest
    among all methods available for that task — filename-only, no image loads."""
    cfg = TASKS[task]
    winners = []
    for image_id in common_ids(task):
        psnrs = {}
        for method, folder in cfg["folders"].items():
            path = _best_file_for_id(folder, cfg["id_pattern"], image_id)
            if path is None:
                psnrs = None
                break
            psnrs[method] = _extract_psnr(path.stem)
        if not psnrs or any(v is None for v in psnrs.values()):
            continue
        if all(psnrs["MDDAir"] > v for k, v in psnrs.items() if k != "MDDAir"):
            winners.append(image_id)
    return winners


def resolve_row(task: str, image_id: str, column_order):
    """Returns dict: key -> (PIL.Image full-size, psnr_or_None), plus 'gt' key."""
    cfg = TASKS[task]
    gt_path = cfg["gt_path"](image_id)
    gt_img = Image.open(gt_path).convert("RGB")

    if cfg.get("synthesize_input"):
        deg_img = _synth_noisy(gt_path, NOISE_SIGMA, seed=hash((task, image_id)) & 0xFFFFFFFF)
    else:
        deg_img = Image.open(cfg["input_path"](image_id)).convert("RGB")

    row = {"deg_img": (deg_img, None), "gt": (gt_img, None)}
    for folder_name in column_order:
        if folder_name not in cfg["folders"]:
            continue
        key = METHOD_KEY_BY_FOLDER_NAME[folder_name]
        path = _best_file_for_id(cfg["folders"][folder_name], cfg["id_pattern"], image_id)
        if path is None:
            return None
        img = Image.open(path).convert("RGB")
        psnr = _extract_psnr(path.stem)
        row[key] = (img, psnr)
    return row


def build_illustration(rows_spec, column_order, out_path: Path,
                        fs_title=30, fs_caption=24, fs_rowlabel=30):
    """rows_spec: list of (task, image_id). column_order: list of folder names
    (subset of PromptIR/InstructIR/DFPIR/MDDAir) in display order, Deg_IMG/GT implicit."""
    resolved_rows = []
    crop_boxes = []
    for task, image_id in rows_spec:
        row = resolve_row(task, image_id, column_order)
        resolved_rows.append((task, row))
        crop_boxes.append(_auto_crop(row["gt"][0]))

    col_keys = ["deg_img"] + [METHOD_KEY_BY_FOLDER_NAME[f] for f in column_order if f in TASKS[rows_spec[0][0]]["folders"] or True]
    # column presence can differ slightly per task (shouldn't, given common_ids already
    # filtered), but guard anyway by using the requested order directly:
    col_keys = ["deg_img"] + [METHOD_KEY_BY_FOLDER_NAME[f] for f in column_order] + ["gt"]

    n_rows = len(resolved_rows)
    n_cols = len(col_keys)
    col_width = 3.4  # inches per panel — generous, print-quality size

    # Fixed, absolute sizes (not a ratio tied to col_width) — large enough to
    # read clearly at this physical size without needing manual pad tuning.
    FS_TITLE = fs_title
    FS_CAPTION = fs_caption
    FS_ROWLABEL = fs_rowlabel
    LW_CROP_BOX = 3.5
    LW_OURS_BORDER = 5.5

    height_ratios = [(b[3] - b[1]) / (b[2] - b[0]) for b in crop_boxes]
    fig_height = col_width * sum(height_ratios)

    # constrained_layout automatically computes spacing to avoid titles/labels
    # colliding with neighboring axes — no manual wspace/hspace/pad tuning needed.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, fig_height),
                              constrained_layout=True,
                              gridspec_kw={"height_ratios": height_ratios})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.01, hspace=0.02)
    if n_rows == 1:
        axes = axes[None, :]

    for i, (task, row) in enumerate(resolved_rows):
        crop_box = crop_boxes[i]
        for j, key in enumerate(col_keys):
            ax = axes[i, j]
            img, psnr = row[key]

            if key == "deg_img":
                ax.imshow(img)
                l, t, r, b = crop_box
                ax.add_patch(Rectangle((l, t), r - l, b - t, edgecolor="red", facecolor="none",
                                        linewidth=LW_CROP_BOX))
            else:
                ax.imshow(img.crop(crop_box))

            ax.set_xticks([])
            ax.set_yticks([])

            is_ours = key == "mddair"
            for spine in ax.spines.values():
                spine.set_visible(is_ours)
                if is_ours:
                    spine.set_edgecolor("red")
                    spine.set_linewidth(LW_OURS_BORDER)

            if psnr is not None:
                ssim = _compute_ssim(img, row["gt"][0])
                ax.set_xlabel(f"{psnr:.2f} dB / {ssim:.4f}", fontsize=FS_CAPTION,
                              fontweight="bold" if is_ours else "normal",
                              color="red" if is_ours else "black",
                              labelpad=14)

            if i == 0:
                label = "Input" if key == "deg_img" else ("GT" if key == "gt" else METHOD_LABELS[key])
                ax.set_title(label, fontsize=FS_TITLE,
                             fontweight="bold" if is_ours else "normal",
                             color="red" if is_ours else "black",
                             pad=14)

            if j == 0:
                row_label = f"{task} ($\\sigma$={NOISE_SIGMA})" if task == "Noise" else task
                ax.set_ylabel(row_label, fontsize=FS_ROWLABEL, fontweight="bold")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out_path}")


def _pick(rng, winners, n):
    """Up to n DISTINCT picks — never reuses an ID. If fewer than n unique
    MDDAir-winning candidates exist, returns fewer than n (caller must shrink
    accordingly) rather than repeating one."""
    if len(winners) < n:
        print(f"  [warn] only {len(winners)} MDDAir-winning IDs available, need {n} — only {len(winners)} unique picks possible")
    return rng.sample(winners, min(n, len(winners)))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    N = 4
    tasks_3deg = ("Noise", "Rain", "Haze")
    tasks_2deg = ("Blur", "Low-light")

    winners_3deg = {}
    for t in tasks_3deg:
        w = mddair_winning_ids(t)
        winners_3deg[t] = w
        print(f"{t}: {len(w)} IDs where MDDAir has the highest PSNR (of {len(common_ids(t))} common)")

    winners_2deg = {}
    for t in tasks_2deg:
        w = mddair_winning_ids(t)
        winners_2deg[t] = w
        print(f"{t}: {len(w)} IDs where MDDAir has the highest PSNR (of {len(common_ids(t))} common)")

    col_order_3deg = ["PromptIR", "InstructIR", "DFPIR", "MDDAir"]
    col_order_2deg = ["InstructIR", "DFPIR", "MDDAir"]

    picks_3deg = {t: _pick(rng, winners_3deg[t], N) for t in tasks_3deg}
    picks_2deg = {t: _pick(rng, winners_2deg[t], N) for t in tasks_2deg}

    # Illustration #1: use a portrait-orientation Noise image (like the subway
    # photo in illustration #3) so the figure's tall row fills the page better.
    # Must not collide with the other 3 already-picked Noise IDs. "021" is
    # excluded — it's a real Urban100 photo with a decorative red frame baked
    # into the source image, not a rendering bug, but it looks bad here.
    BAD_NOISE_IDS = {"021"}
    portrait_noise = [
        id_ for id_ in winners_3deg["Noise"]
        if id_ not in picks_3deg["Noise"][1:]
        and id_ not in BAD_NOISE_IDS
        and Image.open(TASKS["Noise"]["gt_path"](id_)).size[1]
        > Image.open(TASKS["Noise"]["gt_path"](id_)).size[0]
    ]
    if portrait_noise:
        picks_3deg["Noise"][0] = rng.choice(portrait_noise)
    else:
        print("  [warn] no portrait Noise candidates found — illustration 1 keeps its landscape pick")

    # Illustration #3's Rain pick was another mountain-valley scene, visually
    # too similar to illustration #2's. Swap in a clearly different scene
    # (stone bridge over a river) — still an MDDAir-winning image.
    rain_replacement = "055"
    if rain_replacement in winners_3deg["Rain"] and rain_replacement not in picks_3deg["Rain"]:
        picks_3deg["Rain"][2] = rain_replacement

    # 4th blur/low-light illustration: Low-light only has 3 MDDAir-winning
    # images, so this one uses a runner-up (MDDAir 2nd-best, by the smallest
    # margin available) rather than reusing an image already shown.
    lowlight_runnerup = "22"  # MDDAir 28.73 vs DFPIR 29.00 — 2nd place, margin 0.27 dB (79 was already used in the main paper)
    if lowlight_runnerup not in picks_2deg["Low-light"]:
        picks_2deg["Low-light"].append(lowlight_runnerup)

    blur_extra_exclude = set(picks_2deg["Blur"])
    blur_extra = [id_ for id_ in winners_2deg["Blur"] if id_ not in blur_extra_exclude]
    if blur_extra:
        picks_2deg["Blur"].append(rng.choice(blur_extra))

    n_3deg = min(len(picks_3deg[t]) for t in tasks_3deg)
    n_2deg = min(len(picks_2deg[t]) for t in tasks_2deg)
    print(f"Building {n_3deg} '3-deg' illustrations and {n_2deg} 'blur/low-light' illustrations")

    for n in range(n_3deg):
        rows_spec = [(t, picks_3deg[t][n]) for t in tasks_3deg]
        build_illustration(rows_spec, col_order_3deg, OUT_DIR / f"3deg_illustration_{n + 1}.pdf")

    for n in range(n_2deg):
        rows_spec = [(t, picks_2deg[t][n]) for t in tasks_2deg]
        build_illustration(rows_spec, col_order_2deg, OUT_DIR / f"blur_lowlight_illustration_{n + 1}.pdf",
                            fs_title=25, fs_caption=20, fs_rowlabel=25)

    print(f"\nDone. Saved to {OUT_DIR}")
