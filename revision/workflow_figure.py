#!/usr/bin/env python3
"""Workflow figure (revision/outputs/workflow.png, 300 dpi)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "dejavuserif"})

steps = [
    ("Data", "SWaT 2021 session\n25 Jun 2021, 1 Hz\nz-score, 70/30 split"),
    ("Identification", r"$x_{k+1} = A_d x_k + B_d u_k + c$" + "\nleast squares\nLQR gain $K$"),
    ("Attack modes &\nsimulation", "DoS, FDI, bias,\nsensor spoofing\n20 seeds, 600 s"),
    ("Reachability", "zonotope sets\n" + r"bound $\bar{\delta}$ vs." + "\nalarm limits"),
    ("Dependency\ngraph", "weights from $B_d$, $K$\npropagation,\nbetweenness"),
    ("Metrics", "peak, RMS,\nrecovery time,\nresilience R"),
]
fig, ax = plt.subplots(figsize=(12, 2.6))
w, h, gap = 1.9, 1.55, 0.35
for i, (title, body) in enumerate(steps):
    x = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc="#eef3f8", ec="#2b4c6f", lw=1.2))
    ax.text(x + w / 2, h - 0.33, title, ha="center", va="center", fontweight="bold", fontsize=9.5)
    ax.text(x + w / 2, 0.52, body, ha="center", va="center", fontsize=8)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.04, h / 2), xytext=(x + w + 0.04, h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#2b4c6f", lw=1.3))
ax.set_xlim(-0.1, len(steps) * (w + gap) - gap + 0.1)
ax.set_ylim(-0.1, h + 0.1)
ax.axis("off")
fig.savefig(os.path.join(OUT, "workflow.png"), dpi=300, bbox_inches="tight")
print("saved", os.path.join(OUT, "workflow.png"))
