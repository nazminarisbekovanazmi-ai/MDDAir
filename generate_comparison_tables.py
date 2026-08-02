"""
Generate booktabs-style comparison tables for 3-task and 5-task settings.
Style matches academic paper format (toprule / midrule / bottomrule, no vertical lines).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUTPUT_DIR = "output/comparison_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MDDAir results ───────────────────────────────────────────────────────────
MDDA_3D = {
    'haze': (29.96, 0.9602), 'rain': (38.87, 0.9840),
    'n15':  (34.38, 0.9453), 'n25':  (31.87, 0.9138),
    'n50':  (28.59, 0.8473), 'avg':  (32.73, 0.9301),
}
MDDA_5D = {
    'haze': (29.92, 0.9595), 'rain': (38.80, 0.9839),
    'n25':  (31.85, 0.9142), 'blur': (27.62, 0.8498),
    'low':  (23.06, 0.8481), 'avg':  (30.20, 0.9088),
}


def fmt(psnr, ssim):
    return f"{psnr:.2f}/{ssim:.3f}"


def best_per_col(rows, n_cols):
    """Row index (0-based) of highest PSNR in each metric column (skip col 0)."""
    best = {}
    for j in range(1, n_cols):
        psnrs = [float(r[j].split('/')[0]) for r in rows]
        best[j] = int(np.argmax(psnrs))
    return best


def draw_hlines(fig, ax, tbl, n_data_rows, thin_before_rows=None):
    """Draw toprule, midrule, bottomrule and optional thin separators."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()

    all_x0, all_x1 = [], []
    row_top, row_bot = {}, {}

    for (r, c), cell in tbl.get_celld().items():
        b = cell.get_window_extent(renderer)
        if c == 0:
            row_top[r] = b.y1
            row_bot[r] = b.y0
        all_x0.append(b.x0)
        all_x1.append(b.x1)

    x0 = min(all_x0);  x1 = max(all_x1)
    x0f, _ = fig_inv.transform((x0, 0))
    x1f, _ = fig_inv.transform((x1, 0))

    def hline(y_display, lw, color='black'):
        _, yf = fig_inv.transform((0, y_display))
        fig.add_artist(mlines.Line2D(
            [x0f, x1f], [yf, yf],
            transform=fig.transFigure, color=color, lw=lw, clip_on=False))

    hline(row_top[0], 1.5)                      # toprule
    hline(row_bot[0], 0.8)                      # midrule (below header)
    hline(row_bot[n_data_rows], 1.5)            # bottomrule

    if thin_before_rows:
        for r in thin_before_rows:
            hline(row_top[r], 0.5, color='#888888')   # thin gray separator


def draw_table(fig_path, caption, col_headers, rows, our_idx,
               figsize, thin_before_rows=None):
    n_cols = len(col_headers)
    n_rows = len(rows)
    best = best_per_col(rows, n_cols)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.9)

    # Hide all edges
    for cell in tbl.get_celld().values():
        cell.set_edgecolor('white')
        cell.set_facecolor('white')

    # Header row: bold
    for j in range(n_cols):
        tbl[0, j].set_text_props(fontweight='bold', va='center', ha='center')

    # Data rows
    for i in range(1, n_rows + 1):
        is_ours = (i - 1 == our_idx)
        for j in range(n_cols):
            cell = tbl[i, j]
            is_best = (j > 0 and best.get(j) == i - 1)
            bold = is_ours or is_best
            cell.set_text_props(fontweight='bold' if bold else 'normal',
                                ha='left' if j == 0 else 'center')

    draw_hlines(fig, ax, tbl, n_rows, thin_before_rows=thin_before_rows)

    # Caption below the table (paper style)
    fig.text(0.5, 0.01, caption, ha='center', va='bottom',
             fontsize=8.5, style='italic', color='#111111')

    plt.savefig(fig_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")


# ════════════════════════════════════════════════════════════════════════════
# TABLE 1 — 3-task
# ════════════════════════════════════════════════════════════════════════════
cols_3d = ["Methods",
           "Dehazing\non SOTS",
           "Deraining\non Rain100L",
           "Denoising\nσ = 15",
           "Denoising\nσ = 25",
           "Denoising\nσ = 50",
           "Average"]

raw_3d = [
    ("DL [11]",
     (26.92,0.391),(32.62,0.931),(33.05,0.914),(30.41,0.861),(26.90,0.740),(29.98,0.875)),
    ("FDGAN [10]",
     (24.71,0.924),(29.89,0.933),(30.25,0.910),(28.81,0.868),(26.43,0.776),(28.02,0.883)),
    ("MPRNet [56]",
     (25.28,0.954),(33.57,0.954),(33.54,0.927),(30.89,0.880),(27.56,0.779),(30.17,0.899)),
    ("AirNet [21]",
     (27.94,0.962),(34.90,0.967),(33.92,0.933),(31.26,0.888),(28.00,0.797),(31.20,0.910)),
    ("Restormer [57]",
     (30.43,0.975),(36.55,0.975),(33.84,0.931),(31.18,0.885),(27.90,0.790),(31.98,0.911)),
    ("PromptIR [39]",
     (30.58,0.974),(36.37,0.972),(33.98,0.933),(31.31,0.888),(28.06,0.799),(32.06,0.913)),
    ("InstructIR [8]",
     (30.22,0.959),(37.98,0.978),(34.15,0.933),(31.52,0.890),(28.30,0.804),(32.43,0.913)),
    ("DFPIR",
     (31.87,0.980),(38.65,0.982),(34.14,0.935),(31.47,0.893),(28.25,0.806),(32.88,0.919)),
    ("MDDAir (Ours)",
     MDDA_3D['haze'], MDDA_3D['rain'],
     MDDA_3D['n15'],  MDDA_3D['n25'], MDDA_3D['n50'], MDDA_3D['avg']),
]

rows_3d = [[r[0]] + [fmt(p, s) for p, s in r[1:]] for r in raw_3d]

draw_table(
    fig_path=os.path.join(OUTPUT_DIR, "comparison_3task.png"),
    caption="Table 1. Comparison to state-of-the-art on three tasks. "
            "PSNR (dB, ↑) and SSIM (↑). Best results per column are in bold.",
    col_headers=cols_3d,
    rows=rows_3d,
    our_idx=len(rows_3d) - 1,
    figsize=(14, 5),
)


# ════════════════════════════════════════════════════════════════════════════
# TABLE 2 — 5-task
# ════════════════════════════════════════════════════════════════════════════
cols_5d = ["Methods",
           "Dehazing\non SOTS",
           "Deraining\non Rain100L",
           "Denoising\non CBSD68",
           "Deblurring\non GoPro",
           "Low-light\non LOL",
           "Average"]

raw_5d = [
    # General image restorers (*)
    ("DGUNet* [33]",
     (24.78,0.940),(36.62,0.971),(31.10,0.883),(27.25,0.837),(21.87,0.823),(28.32,0.891)),
    ("SwinIR* [25]",
     (21.50,0.891),(30.78,0.923),(30.59,0.868),(24.52,0.773),(17.81,0.723),(25.04,0.835)),
    ("Restormer* [57]",
     (24.09,0.927),(34.81,0.962),(31.49,0.884),(27.22,0.829),(20.41,0.806),(27.60,0.881)),
    ("NAFNet* [3]",
     (25.23,0.939),(35.56,0.967),(31.02,0.883),(26.53,0.808),(20.49,0.809),(27.76,0.881)),
    # All-in-one methods
    ("DL [11]",
     (20.54,0.826),(21.96,0.762),(23.09,0.745),(19.86,0.672),(19.83,0.712),(21.05,0.743)),
    ("Transweather [46]",
     (21.32,0.885),(29.43,0.905),(29.00,0.841),(25.12,0.757),(21.21,0.792),(25.22,0.836)),
    ("TAPE [28]",
     (22.16,0.861),(29.67,0.904),(30.18,0.855),(24.47,0.763),(18.97,0.621),(25.09,0.801)),
    ("AirNet [21]",
     (21.04,0.884),(32.98,0.951),(30.91,0.882),(24.35,0.781),(18.18,0.735),(25.49,0.846)),
    ("IDR [59]",
     (25.24,0.943),(35.63,0.965),(31.60,0.887),(27.87,0.846),(21.34,0.826),(28.34,0.893)),
    ("InstructIR [8]",
     (27.10,0.956),(36.84,0.973),(31.40,0.887),(29.40,0.886),(23.00,0.836),(29.55,0.907)),
    ("DFPIR",
     (31.64,0.979),(37.62,0.978),(31.29,0.889),(28.82,0.873),(23.82,0.843),(30.64,0.913)),
    ("MDDAir (Ours)",
     MDDA_5D['haze'], MDDA_5D['rain'],
     MDDA_5D['n25'],  MDDA_5D['blur'], MDDA_5D['low'], MDDA_5D['avg']),
]

rows_5d = [[r[0]] + [fmt(p, s) for p, s in r[1:]] for r in raw_5d]

draw_table(
    fig_path=os.path.join(OUTPUT_DIR, "comparison_5task.png"),
    caption="Table 2. Comparison to state-of-the-art on five tasks. "
            "PSNR (dB, ↑) and SSIM (↑). * denotes general image restoration methods. "
            "Best results per column are in bold.",
    col_headers=cols_5d,
    rows=rows_5d,
    our_idx=len(rows_5d) - 1,
    figsize=(15, 7),
    thin_before_rows=[5],   # thin gray line between specialized (*) and all-in-one methods
)

# ════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Ablation study (3-task)
# Fill in each variant's results after training + testing finishes.
# Leave a tuple as None to show "—" placeholder.
# ════════════════════════════════════════════════════════════════════════════

ABLATION_3D = {
    # key: (haze, rain, n15, n25, n50, avg)  — set to None until tested
    'no_deg_estimator': None,   # fill after ablation\no_deg_est run
    'no_film':          None,   # fill after ablation\no_film run
    'no_spatial_attn':  None,   # fill after ablation\no_sp_attn run
}

VARIANT_LABELS = {
    'no_deg_estimator': 'w/o Deg. Estimator',
    'no_film':          'w/o FiLM',
    'no_spatial_attn':  'w/o Spatial Attn.',
}

cols_abl = ["Method",
            "Dehazing\non SOTS",
            "Deraining\non Rain100L",
            "Denoising\nσ=15",
            "Denoising\nσ=25",
            "Denoising\nσ=50",
            "Average"]


def fmt_or_dash(val):
    if val is None:
        return "—"
    p, s = val
    return f"{p:.2f}/{s:.3f}"


raw_abl = [("MDDAir (Full)",
            MDDA_3D['haze'], MDDA_3D['rain'],
            MDDA_3D['n15'],  MDDA_3D['n25'], MDDA_3D['n50'], MDDA_3D['avg'])]

for key in ['no_deg_estimator', 'no_film', 'no_spatial_attn']:
    res = ABLATION_3D[key]
    if res is None:
        row = [VARIANT_LABELS[key]] + ["—"] * 6
    else:
        haze, rain, n15, n25, n50, avg = res
        row = [VARIANT_LABELS[key],
               fmt(haze[0], haze[1]), fmt(rain[0], rain[1]),
               fmt(n15[0],  n15[1]),  fmt(n25[0],  n25[1]),
               fmt(n50[0],  n50[1]),  fmt(avg[0],  avg[1])]
    raw_abl.append(row)

# Convert full model row to same string format
rows_abl = [[raw_abl[0][0]] + [fmt(p, s) for p, s in raw_abl[0][1:]]] + raw_abl[1:]

draw_table(
    fig_path=os.path.join(OUTPUT_DIR, "ablation_3task.png"),
    caption="Table 3. Ablation study on 3-task restoration. "
            "PSNR (dB, ↑) and SSIM (↑). Best results per column are in bold.",
    col_headers=cols_abl,
    rows=rows_abl,
    our_idx=0,          # full model is the first (baseline) row
    figsize=(14, 3.5),
    thin_before_rows=None,
)

print("\nDone. Tables saved to:", OUTPUT_DIR)
