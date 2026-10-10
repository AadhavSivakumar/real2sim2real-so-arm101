"""Plot the COLMAP reconstruction: sparse points + camera centers/view directions per video.

Usage: python plot_sfm.py <sparse_model_dir> <out.png>
"Down" is the average camera viewing direction (the phone mostly looks down at the desk).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

rec = pycolmap.Reconstruction(sys.argv[1])
pts = np.array([p.xyz for p in rec.points3D.values()])
rgb = np.array([p.color for p in rec.points3D.values()]) / 255.0

# World frame: z = up (opposite the mean viewing direction), x = main spread of the camera path.
ims = list(rec.images.values())
fwds = np.array([im.cam_from_world().rotation.matrix()[2] for im in ims])
ctrs = np.array([im.projection_center() for im in ims])
z = -fwds.mean(0); z /= np.linalg.norm(z)
flat = (ctrs - ctrs.mean(0)) - np.outer((ctrs - ctrs.mean(0)) @ z, z)
x = np.linalg.svd(flat, full_matrices=False)[2][0]; x -= (x @ z) * z; x /= np.linalg.norm(x)
vt = np.stack([x, np.cross(z, x), z])
center = np.median(pts, axis=0)
to_pc = lambda x: (x - center) @ vt.T

P = to_pc(pts)
cams = {}
for im in sorted(rec.images.values(), key=lambda im: im.name):
    c = im.projection_center()
    fwd = im.cam_from_world().rotation.matrix()[2]  # camera +z (viewing direction) in world coords
    cams.setdefault(im.name.split("_")[0], []).append((to_pc(c), fwd @ vt.T))

lim = np.percentile(np.abs(P), 90, axis=0) * 1.3
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
colors = {"v3495": "tab:red", "v3496": "tab:blue"}
for ax, (i, j), title in [(axes[0], (0, 1), "Top-down (looking along mean camera direction)"), (axes[1], (0, 2), "Side view (vertical = up)")]:
    ax.scatter(P[:, i], P[:, j], c=rgb, s=1.5, alpha=0.8)
    for vid, lst in cams.items():
        C = np.array([c for c, _ in lst]); F = np.array([f for _, f in lst])
        ax.plot(C[:, i], C[:, j], "-", color=colors.get(vid, "k"), lw=1, alpha=0.6)
        ax.quiver(C[::3, i], C[::3, j], F[::3, i], F[::3, j], color=colors.get(vid, "k"),
                  scale=30, width=0.003, label=f"{vid}: {len(lst)} cameras")
    ax.set_xlim(-lim[i], lim[i]); ax.set_ylim(-lim[j], lim[j]); ax.set_aspect("equal")
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
fig.suptitle(f"COLMAP reconstruction: {rec.num_reg_images()} cameras, {len(pts)} points, "
             f"mean reprojection error {rec.compute_mean_reprojection_error():.2f}px")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print("saved", sys.argv[2])
