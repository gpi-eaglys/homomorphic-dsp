import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
FPATH_PTH  = os.path.join(REPO_DIR, "assets/mdl/PANN/Cnn14_mAP%3D0.431.pth")

ckpt = torch.load(FPATH_PTH, map_location="cpu", weights_only=False)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FPATH_PNG = os.path.join(REPO_DIR, "assets/mdl/PANN/Cnn14_arch.png")

layers = [
    ("STFT",           "spectrogram\nextractor",  1,    513),
    ("MelW",           "logmel\nextractor",       513,   64),
    ("BN0",            "batch norm",              64,    64),
    ("Conv block 1",   "Conv 1→64\nConv 64→64",   1,    64),
    ("Conv block 2",   "Conv 64→128\nConv 128→128", 64, 128),
    ("Conv block 3",   "Conv 128→256\nConv 256→256", 128, 256),
    ("Conv block 4",   "Conv 256→512\nConv 512→512", 256, 512),
    ("Conv block 5",   "Conv 512→1024\nConv 1024→1024", 512, 1024),
    ("Conv block 6",   "Conv 1024→2048\nConv 2048→2048", 1024, 2048),
    ("FC1",            "Linear\n2048→2048",       2048, 2048),
    ("fc_audioset",    "Linear\n2048→527",        2048,  527),
]

fig, ax = plt.subplots(figsize=(4, 14))
ax.set_xlim(0, 4)
ax.set_ylim(-0.5, len(layers) + 0.5)
ax.axis("off")
ax.set_title("PANN Cnn14 Architecture", fontsize=13, fontweight="bold", pad=12)

colors = ["#aec6e8", "#aec6e8", "#d4edda",
          "#ffd9a0", "#ffc87a", "#ffb347", "#ff9a1f",
          "#ff7f00", "#e86c00", "#b5ead7", "#c9b1e8"]

for i, (name, detail, ch_in, ch_out) in enumerate(reversed(layers)):
    y = i
    color = colors[len(layers) - 1 - i]
    rect = mpatches.FancyBboxPatch((0.2, y - 0.38), 3.6, 0.76,
                                    boxstyle="round,pad=0.04",
                                    facecolor=color, edgecolor="#555", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(2.0, y + 0.12, name, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(2.0, y - 0.18, detail, ha="center", va="center", fontsize=7, color="#333",
            linespacing=1.3)
    ax.text(3.72, y, f"→{ch_out}", ha="right", va="center", fontsize=7, color="#555")

fig.tight_layout()
fig.savefig(FPATH_PNG, dpi=130, bbox_inches="tight")
print(f"Saved: {FPATH_PNG}")
