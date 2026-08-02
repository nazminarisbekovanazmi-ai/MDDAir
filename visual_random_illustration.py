"""Blind qualitative comparison: 20 randomly-picked images (5 each from Noise/Urban100,
Blur, Rain, Low-light), copied into per-image folders (Img-1, Img-2, ...) with files
named by a fixed method->number mapping: 1=MDDAir, 2=DFPIR, 3=InstructIR, 4=PromptIR.
No method names, no PSNR/SSIM, no figure — just the raw images.

Only real results from these four methods are used (no substitutes). Per task, only
the methods that actually have matching data get a numbered file in that image's
folder — the others are simply absent:
  - Rain:              MDDAir, DFPIR, InstructIR, PromptIR (all four match)
  - Noise (Urban100):  MDDAir, InstructIR only (DFPIR/PromptIR only ever saved
                        BSD68 noise results, never Urban100, in these curated folders)
  - Blur:               MDDAir, DFPIR, InstructIR only (PromptIR has no deblur results)
  - Low-light:          MDDAir, DFPIR, InstructIR only (PromptIR has no lowlight results)
"""
import random
import re
import shutil
from pathlib import Path

QUAL_BASE = Path(r"C:\Users\Nazmi\Desktop\MDDAir\Qualitive Results")
OUT_DIR = QUAL_BASE / "Visual Random Illustration"

METHOD_NUMBER = {"MDDAir": 1, "DFPIR": 2, "InstructIR": 3, "PromptIR": 4}
IMG_EXTS = (".png", ".jpg", ".jpeg")
SEED = 42  # change/remove for a different random draw each run
PER_TASK = 5  # 4 tasks x 5 = 20

TASKS = {
    "rain": {
        "folders": {
            "MDDAir": QUAL_BASE / "MDDAir" / "derain",
            "DFPIR": QUAL_BASE / "DFPIR" / "derain",
            "InstructIR": QUAL_BASE / "InstructIR" / "Rain100L",
            "PromptIR": QUAL_BASE / "PromptIR" / "derain",
        },
        "id_pattern": re.compile(r"(?:rain|norain)-(\d+)"),
    },
    "noise_urban100": {
        "folders": {
            "MDDAir": QUAL_BASE / "MDDAir" / "denoise_25",
            "InstructIR": QUAL_BASE / "InstructIR" / "Urban100_25",
        },
        "id_pattern": re.compile(r"img_(\d+)_SRF", re.IGNORECASE),
    },
    "blur": {
        "folders": {
            "MDDAir": QUAL_BASE / "MDDAir" / "deblur",
            "DFPIR": QUAL_BASE / "DFPIR" / "deblur",
            "InstructIR": QUAL_BASE / "InstructIR" / "GoPro",
        },
        "id_pattern": re.compile(r"^(\d+)[_.]"),
    },
    "lowlight": {
        "folders": {
            "MDDAir": QUAL_BASE / "MDDAir" / "lowlight",
            "DFPIR": QUAL_BASE / "DFPIR" / "lowlight",
            "InstructIR": QUAL_BASE / "InstructIR" / "LOL",
        },
        "id_pattern": re.compile(r"^(\d+)[_.]"),
    },
}


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
    """Among files matching target_id, prefer the one with the highest embedded
    PSNR (handles methods that accumulated multiple epochs' worth of saves)."""
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


def gather_common_ids(task: str):
    cfg = TASKS[task]
    id_sets = [_list_ids(folder, cfg["id_pattern"]) for folder in cfg["folders"].values()]
    return sorted(set.intersection(*id_sets))


def copy_one_set(task: str, image_id: str, img_dir: Path):
    cfg = TASKS[task]
    img_dir.mkdir(parents=True, exist_ok=True)
    for method, folder in cfg["folders"].items():
        path = _best_file_for_id(folder, cfg["id_pattern"], image_id)
        if path is None:
            print(f"  [warn] {task} {image_id}: no file for {method}, skipping that slot")
            continue
        n = METHOD_NUMBER[method]
        shutil.copyfile(path, img_dir / f"{n}{path.suffix.lower()}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    picks = []  # list of (task, id)
    for task in TASKS:
        common = gather_common_ids(task)
        print(f"{task}: {len(common)} IDs common to {list(TASKS[task]['folders'])}")
        chosen = rng.sample(common, min(PER_TASK, len(common)))
        picks.extend((task, id_) for id_ in chosen)

    rng.shuffle(picks)

    for i, (task, image_id) in enumerate(picks, start=1):
        img_dir = OUT_DIR / f"Img-{i}"
        copy_one_set(task, image_id, img_dir)
        print(f"Saved {img_dir.name}/ ({task} {image_id})")

    print(f"\nDone: {len(picks)} image sets saved under {OUT_DIR}")
