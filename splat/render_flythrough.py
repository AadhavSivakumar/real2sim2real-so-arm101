"""Render a smooth flythrough of a trained gsplat checkpoint along the captured camera path.

Usage: python render_flythrough.py <data_dir> <ckpt.pt> <out.mp4>
Each video's camera poses are smoothed with a spline (no interpolation *between* videos,
which would fly through unobserved space).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gsplat" / "examples"))

import imageio
import numpy as np
import torch
import tqdm
from datasets.colmap import Parser
from datasets.traj import generate_interpolated_path
from gsplat.rendering import rasterization

data_dir, ckpt_path, out_path = sys.argv[1:4]
device = "cuda"

parser = Parser(data_dir=data_dir, factor=1, normalize=True, test_every=8)
splats = torch.load(ckpt_path, map_location=device, weights_only=False)["splats"]
means, quats = splats["means"], splats["quats"]
scales, opacities = torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"])
colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
sh_degree = int(np.sqrt(colors.shape[1]) - 1)

names = np.array(parser.image_names)
segments = []
for prefix in ["v3496", "v3495"]:  # long orbit first, then the short clip
    idx = np.where(np.char.startswith(names, prefix))[0]
    idx = idx[np.argsort(names[idx])][2:-2]
    path = generate_interpolated_path(parser.camtoworlds[idx][::2, :3, :], n_interp=3)
    segments.append((prefix, idx, path))

writer = imageio.get_writer(out_path, fps=30, quality=8, macro_block_size=1)
for prefix, idx, path in segments:
    cam_id = parser.camera_ids[idx[0]]
    K = torch.from_numpy(parser.Ks_dict[cam_id]).float().to(device)
    w, h = parser.imsize_dict[cam_id]
    w, h = w - w % 2, h - h % 2  # H.264 needs even dimensions
    c2w = np.concatenate([path, np.tile([[[0, 0, 0, 1.0]]], (len(path), 1, 1))], axis=1)
    viewmats = torch.linalg.inv(torch.from_numpy(c2w).float().to(device))
    for i in tqdm.trange(len(viewmats), desc=f"rendering {prefix}"):
        with torch.no_grad():
            img, _, _ = rasterization(means, quats, scales, opacities, colors,
                                      viewmats[i : i + 1], K[None], w, h, sh_degree=sh_degree)
        writer.append_data((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    print(f"{prefix}: {len(idx)} captured poses -> {len(path)} rendered frames ({w}x{h})")
writer.close()
print("saved", out_path)
