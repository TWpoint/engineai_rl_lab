"""Project captured policy latents to 3D and draw time-ordered trajectories."""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("inputs", nargs="+", help="NPZ files produced by play.py --latent_output")
parser.add_argument("--output", required=True, help="Output PNG path")
parser.add_argument("--title", default="Policy Latent Space")
parser.add_argument("--stride", type=int, default=1, help="Plot every Nth sample")
parser.add_argument("--sphere", action="store_true", help="Normalize projected points onto a common unit sphere")
args = parser.parse_args()

records = []
for filename in args.inputs:
    path = pathlib.Path(filename)
    with np.load(path) as data:
        latent = np.asarray(data["latent"], dtype=np.float64)
        motion = str(data.get("motion_file", path.stem))
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError(f"Expected at least two latent vectors in {path}, got {latent.shape}")
    records.append((path, pathlib.Path(motion).stem, latent))

all_latents = np.concatenate([record[2] for record in records], axis=0)
mean = all_latents.mean(axis=0, keepdims=True)
_, singular_values, components = np.linalg.svd(all_latents - mean, full_matrices=False)
basis = components[:3].T
explained = singular_values[:3] ** 2 / np.maximum((singular_values**2).sum(), np.finfo(float).eps)

fig = plt.figure(figsize=(9, 8), dpi=180)
ax = fig.add_subplot(111, projection="3d")
colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(records)))
for color, (_, label, latent) in zip(colors, records, strict=True):
    xyz = (latent - mean) @ basis
    if args.sphere:
        xyz /= np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), np.finfo(float).eps)
    xyz = xyz[:: args.stride]
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=1.8, alpha=0.9, label=label)
    ax.scatter(*xyz[0], color=color, s=24, marker="o")
    ax.scatter(*xyz[-1], color=color, s=30, marker="x")

ax.set_title(args.title, pad=18, fontsize=15)
ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
ax.set_zlabel(f"PC3 ({explained[2]:.1%})")
ax.grid(True, alpha=0.25)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.85)
fig.tight_layout()
output = pathlib.Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
print(f"Saved latent-space plot to {output}")
