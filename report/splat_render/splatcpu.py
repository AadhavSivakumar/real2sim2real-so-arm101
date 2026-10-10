"""CPU-only 3DGS renderer (torch, tile-based EWA splatting, front-to-back alpha compositing), gsplat-compatible maths.

World frame = robot/sim desk frame (Z up, metres). PLY -> world: x_w = s R x_ply + t (R from SPLAT_QUAT, xyzw).
Cameras use the OpenCV optical convention (x right, y down, z forward); R_cw is camera-to-world.
Optional opaque desk-top quad (the sim's desk_top_hr.usda: desk_albedo texture on the footprint polygon at z=0.0005),
depth-tested per pixel against the Gaussians, exactly like the sim composes splat + desk.
"""
import json
import os

import cv2
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial.transform import Rotation

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "10")))
C0 = 0.28209479177387814
DESK_POLY = np.array([(-0.12, 0.60), (-0.12, -0.226), (0.386, -0.164), (0.339, 0.60)])
TEX_QUAD = (-0.20, 0.54, -0.60, 0.60)  # x0 x1 y0 y1 the texture covers (s along -y, t along +x)


def sh_bases(d):
    """Real SH bases up to degree 3 (gsplat _eval_sh_bases_fast ordering). d: (N,3) unit."""
    x, y, z = d.unbind(-1)
    r = torch.empty(d.shape[0], 16, dtype=d.dtype)
    r[:, 0] = 0.2820947917738781
    a = -0.48860251190292
    r[:, 2] = -a * z
    r[:, 3] = a * x
    r[:, 1] = a * y
    z2 = z * z
    b = -1.092548430592079 * z
    a = 0.5462742152960395
    c1 = x * x - y * y
    s1 = 2 * x * y
    r[:, 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    r[:, 7] = b * x
    r[:, 5] = b * y
    r[:, 8] = a * c1
    r[:, 4] = a * s1
    c = -2.285228997322329 * z2 + 0.4570457994644658
    b = 1.445305721320277 * z
    a = -0.5900435899266435
    c2 = x * c1 - y * s1
    s2 = x * s1 + y * c1
    r[:, 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    r[:, 13] = c * x
    r[:, 11] = c * y
    r[:, 14] = b * c1
    r[:, 10] = b * s1
    r[:, 15] = a * c2
    r[:, 9] = a * s2
    return r


class Splat:
    def __init__(self, ply, pose_json, gain_mode="baked"):
        """gain_mode: 'baked' = PLY as-is (export_splat.py already baked color_gain into the SH);
        'undo' = divide the gain back out (phone-video colours); 'double' = multiply by color_gain again."""
        v = PlyData.read(ply)["vertex"].data
        pose = json.load(open(pose_json))
        s = float(pose["SPLAT_SCALE"])
        R = Rotation.from_quat(pose["SPLAT_QUAT"]).as_matrix()  # xyzw (scipy), as in export_splat.py
        t = np.asarray(pose["SPLAT_POS"], float)
        g = np.asarray(pose["color_gain"], float)
        xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1).astype(np.float64)
        rest_names = sorted([n for n in v.dtype.names if n.startswith("f_rest_")], key=lambda n: int(n.split("_")[-1]))
        rest = np.stack([v[n] for n in rest_names], -1).astype(np.float64).reshape(-1, 3, len(rest_names) // 3)
        if gain_mode != "baked":
            gg = 1.0 / g if gain_mode == "undo" else g
            dc = (gg * (0.5 + C0 * dc) - 0.5) / C0
            rest = rest * gg[None, :, None]
        sh = np.concatenate([dc[:, None, :], rest.transpose(0, 2, 1)], 1)  # (N,16,3)
        op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
        sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))
        q = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)  # wxyz (3DGS)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        Rg = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        M = (s * R)[None] @ Rg * sc[:, None, :]  # world  = M M^T
        cov = M @ M.transpose(0, 2, 1)
        self.means = torch.tensor(xyz @ (s * R).T + t, dtype=torch.float32)
        self.cov = torch.tensor(cov, dtype=torch.float32)
        self.sh = torch.tensor(sh, dtype=torch.float32)
        self.op = torch.tensor(op, dtype=torch.float32)
        self.R_ply = torch.tensor(R, dtype=torch.float32)  # world dir -> ply dir: d @ R
        self.n = len(xyz)
        self.scale_max = torch.tensor(sc.max(1) * s, dtype=torch.float32)

    def subset(self, keep):
        o = object.__new__(Splat)
        for k in ("means", "cov", "sh", "op", "scale_max"):
            setattr(o, k, getattr(self, k)[keep])
        o.R_ply = self.R_ply
        o.n = int(o.means.shape[0])
        return o


class Desk:
    def __init__(self, tex_path, z=0.0005):
        img = cv2.imread(tex_path, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
        self.levels = [img]
        while self.levels[-1].shape[1] > 300:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        self.z = z

    def layer(self, C, R_cw, K, W, H):
        """Per-pixel desk depth (camera z, inf off-desk) and colour (H,W,3)."""
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        d = rays_c @ R_cw.T
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = (self.z - C[2]) / d[..., 2]
        X = C + tz[..., None] * d
        x, y = X[..., 0], X[..., 1]
        inside = tz > 0
        P = DESK_POLY
        for i in range(len(P)):  # convex polygon, CCW? test sign consistency
            a, b = P[i], P[(i + 1) % len(P)]
            cr = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            inside &= cr <= 0 if self._cw else cr >= 0
        x0, x1, y0, y1 = TEX_QUAD
        s_ = (y1 - y) / (y1 - y0)
        t_ = (x - x0) / (x1 - x0)
        # per-pixel trilinear mip sampling: lod from the texel footprint (texels of level 0 per screen pixel)
        img0 = self.levels[0]
        s_ = np.where(inside, s_, 0.5)
        t_ = np.where(inside, t_, 0.5)
        col0, row0 = s_ * img0.shape[1], (1 - t_) * img0.shape[0]
        fx_ = np.hypot(np.gradient(col0, axis=1), np.gradient(row0, axis=1))
        fy_ = np.hypot(np.gradient(col0, axis=0), np.gradient(row0, axis=0))
        lod = np.clip(np.log2(np.maximum(np.maximum(fx_, fy_), 1.0)), 0, len(self.levels) - 1)
        col = np.zeros((H, W, 3), np.float32)
        for li, img in enumerate(self.levels):
            wl = np.clip(1.0 - np.abs(lod - li), 0, 1)
            if not (wl > 0).any():
                continue
            mapx = (s_ * img.shape[1] - 0.5).astype(np.float32)
            mapy = ((1 - t_) * img.shape[0] - 0.5).astype(np.float32)
            col += wl[..., None].astype(np.float32) * cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                                                               borderMode=cv2.BORDER_REPLICATE)
        depth = np.where(inside, tz, np.inf).astype(np.float32)
        return depth, col, inside

    @property
    def _cw(self):
        P = DESK_POLY
        area = 0.5 * sum(P[i, 0] * P[(i + 1) % len(P), 1] - P[(i + 1) % len(P), 0] * P[i, 1] for i in range(len(P)))
        return area < 0


def look_at(C, target, up=(0, 0, 1)):
    C, target, up = (np.asarray(a, float) for a in (C, target, up))
    f = target - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], 1)  # columns: cam x (right), y (down), z (forward) in world


@torch.inference_mode()
def render(sp, C, R_cw, K, W, H, desk=None, bg=(0.0, 0.0, 0.0), tile=16, near=0.02, batch=24, chunk=256,
           return_stats=False, near_fade=None):
    C_t = torch.tensor(C, dtype=torch.float32)
    Rcw = torch.tensor(R_cw, dtype=torch.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    pc = (sp.means - C_t) @ Rcw
    z = pc[:, 2]
    m = z > near
    # coarse frustum cull with generous margin
    tanx, tany = 0.5 * W / fx, 0.5 * H / fy
    lim_xp, lim_xn = (W - cx) / fx + 0.3 * tanx, cx / fx + 0.3 * tanx
    lim_yp, lim_yn = (H - cy) / fy + 0.3 * tany, cy / fy + 0.3 * tany
    zc = z.clamp(min=near)
    xr, yr = pc[:, 0] / zc, pc[:, 1] / zc
    m &= (xr > -lim_xn - 1.0) & (xr < lim_xp + 1.0) & (yr > -lim_yn - 1.0) & (yr < lim_yp + 1.0)
    idx = torch.nonzero(m).squeeze(1)
    pc, z = pc[idx], z[idx]
    tx = z * (pc[:, 0] / z).clamp(-lim_xn, lim_xp)
    ty = z * (pc[:, 1] / z).clamp(-lim_yn, lim_yp)
    # J (2x3)
    J = torch.zeros(len(idx), 2, 3)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * tx / z ** 2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * ty / z ** 2
    cov_c = Rcw.T @ sp.cov[idx] @ Rcw
    cov2 = J @ cov_c @ J.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    ok = det > 0
    inv = 1.0 / det.clamp(min=1e-12)
    ca, cb, cc = c * inv, -b * inv, a * inv
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    rad = torch.ceil(3.0 * torch.sqrt(lam))
    u = fx * pc[:, 0] / z + cx
    v = fy * pc[:, 1] / z + cy
    ok &= (rad > 0) & (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H)
    sel = torch.nonzero(ok).squeeze(1)
    gi = idx[sel]
    u, v, z, rad = u[sel], v[sel], z[sel], rad[sel]
    ca, cb, cc = ca[sel], cb[sel], cc[sel]
    op = sp.op[gi]
    if near_fade is not None:  # smooth opacity fade-in with camera depth (no popping of near-camera floaters)
        a0, a1 = near_fade
        x_ = ((z - a0) / (a1 - a0)).clamp(0, 1)
        op = op * x_ * x_ * (3 - 2 * x_)
    # colours (SH deg 3), view dir in PLY frame
    d = sp.means[gi] - C_t
    d = d / d.norm(dim=1, keepdim=True)
    basis = sh_bases(d @ sp.R_ply)
    rgb = (basis[:, :, None] * sp.sh[gi]).sum(1) + 0.5
    rgb = rgb.clamp(min=0.0)
    # tiles
    TW, TH = (W + tile - 1) // tile, (H + tile - 1) // tile
    tx0 = ((u - rad) / tile).floor().clamp(0, TW - 1).long()
    tx1 = ((u + rad) / tile).floor().clamp(0, TW - 1).long()
    ty0 = ((v - rad) / tile).floor().clamp(0, TH - 1).long()
    ty1 = ((v + rad) / tile).floor().clamp(0, TH - 1).long()
    nw = tx1 - tx0 + 1
    cnt = nw * (ty1 - ty0 + 1)
    order = torch.argsort(z)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    gid = torch.repeat_interleave(torch.arange(len(cnt)), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(len(gid)) - start[gid]
    tile_id = (ty0[gid] + k // nw[gid]) * TW + (tx0[gid] + k % nw[gid])
    key = tile_id * len(order) + rank[gid]
    key, perm = torch.sort(key)
    gsorted = gid[perm]
    tcount = torch.bincount(tile_id, minlength=TW * TH)
    tstart = torch.cumsum(tcount, 0) - tcount
    # desk layer
    if desk is not None:
        ddepth, dcol, dmask = desk.layer(np.asarray(C, float), np.asarray(R_cw, float), K, W, H)
    else:
        ddepth = np.full((H, W), np.inf, np.float32)
        dcol = np.zeros((H, W, 3), np.float32)
        dmask = np.zeros((H, W), bool)
    # pad image to tile multiple
    Hp, Wp = TH * tile, TW * tile
    dd = torch.full((Hp, Wp), float("inf"))
    dd[:H, :W] = torch.from_numpy(ddepth)
    # per-tile pixel coords and desk depth, (T, P)
    ly, lx = torch.meshgrid(torch.arange(tile), torch.arange(tile), indexing="ij")
    lx, ly = lx.reshape(-1).float(), ly.reshape(-1).float()
    tyy, txx = torch.meshgrid(torch.arange(TH), torch.arange(TW), indexing="ij")
    tyy, txx = tyy.reshape(-1), txx.reshape(-1)
    out = torch.zeros(TH * TW, tile * tile, 3)
    Tfin = torch.ones(TH * TW, tile * tile)
    dd_t = dd.reshape(TH, tile, TW, tile).permute(0, 2, 1, 3).reshape(TH * TW, tile * tile)
    params = torch.stack([u, v, ca, cb, cc, op, z], 1)
    active = torch.nonzero(tcount > 0).squeeze(1)
    active = active[torch.argsort(tcount[active], descending=True)]
    work = 0
    for bi in range(0, len(active), batch):
        tb = active[bi:bi + batch]
        n_b = tcount[tb]
        s_b = tstart[tb]
        px = (txx[tb] * tile)[:, None].float() + lx[None] + 0.5  # (B,P)
        py = (tyy[tb] * tile)[:, None].float() + ly[None] + 0.5
        ddb = dd_t[tb][:, :, None]
        T = torch.ones(len(tb), tile * tile)
        col = torch.zeros(len(tb), tile * tile, 3)
        L = int(n_b.max())
        for c0 in range(0, L, chunk):
            ar = torch.arange(c0, min(c0 + chunk, L))
            valid = ar[None] < n_b[:, None]
            pos = (s_b[:, None] + ar[None]).clamp(max=len(gsorted) - 1)
            g = gsorted[pos]  # (B,Lc)
            pg = params[g]  # (B,Lc,7)
            dx = px[:, :, None] - pg[:, None, :, 0]
            dy = py[:, :, None] - pg[:, None, :, 1]
            power = -0.5 * (pg[:, None, :, 2] * dx * dx + pg[:, None, :, 4] * dy * dy) - pg[:, None, :, 3] * dx * dy
            alpha = (pg[:, None, :, 5] * torch.exp(power)).clamp(max=0.999)
            alpha = alpha * ((power <= 0) & (alpha >= 1.0 / 255) & valid[:, None, :] &
                             (pg[:, None, :, 6] < ddb)).float()
            om = 1.0 - alpha
            cp = torch.cumprod(om, -1)
            Tex = T[:, :, None] * torch.cat([torch.ones_like(cp[..., :1]), cp[..., :-1]], -1)
            w = alpha * Tex * (Tex > 1e-4).float()
            col += torch.einsum("bpl,blc->bpc", w, rgb[g])
            T = T * cp[..., -1]
            work += alpha.numel()
            if float(T.max()) < 1e-4:
                break
        out[tb] = col
        Tfin[tb] = T
    img = out.reshape(TH, TW, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(Hp, Wp, 3)[:H, :W]
    Tf = Tfin.reshape(TH, TW, tile, tile).permute(0, 2, 1, 3).reshape(Hp, Wp)[:H, :W]
    under = torch.where(torch.from_numpy(dmask)[..., None], torch.from_numpy(dcol), torch.tensor(bg, dtype=torch.float32))
    img = img + Tf[..., None] * under
    img = img.clamp(0, 1).numpy()
    if return_stats:
        return img, dict(n_vis=len(gi), pairs=len(gid), work=work, alpha=(1 - Tf).numpy(), desk=dmask)
    return img


def to_bgr8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1]
