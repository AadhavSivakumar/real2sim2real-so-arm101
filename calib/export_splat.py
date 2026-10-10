"""Crop a trained gsplat PLY for the sim and write its pose in the desk frame (align_splat.py / refine_splat_align.py
-> SfM->desk transform), plus the desk-top quad the sim draws instead of the splat's desk.

gsplat trains in a normalised copy of the COLMAP frame (Parser(normalize=True).transform), so the PLY -> desk transform
is (SfM -> desk) o (normalised -> SfM). What is dropped from the splat:
  desk      everything over the desk footprint from below the top (the glossy wood is reconstructed 5-15 cm *below* its
            surface, mirror-image style, so its depth is wrong from close by) up to --desk_z[1]: the desk top and
            whatever stood on it (robot, cup, cube). The sim draws the desk top itself: OUT/desk_top.usda, the real
            desk texture (assets/real2sim/desk_albedo.png, overhead camera) on a quad clipped to the same footprint.
  work      the workspace cylinder around the robot base (parts of the arm/clamp beyond the desk edge)
  mount     with --calib: Gaussians just in front of the overhead camera (its own boom; the real camera never sees it)
  stray     far below the floor
The colour gain from refine_splat_align.py (the overhead camera's exposure) is baked into the SH coefficients.
Writes OUT/background.ply (still in the PLY frame), OUT/desk_top.usda, OUT/background_root.usda and OUT/splat_pose.json
(SPLAT_POS / SPLAT_QUAT (xyzw) / SPLAT_SCALE for env_cfg's background prim). Then, on an Isaac box, convert the PLY with
the omni.kit.converter.gsplat extension and point calib/data/splat.json (SO101_SPLAT=1) at the wrapper:
  PYTHONPATH=<isaacsim>/extscache/omni.kit.converter.gsplat-*/pip_prebundle python -m usd_convert_gsplat \
      -i OUT/background.ply -o OUT/background.usdc -n background --up-axis Z
  splat.json: SPLAT_USD = OUT/background_root.usda, SPLAT_POS / SPLAT_QUAT / SPLAT_SCALE from splat_pose.json

  ~/gs/bin/python calib/export_splat.py --scene SCENE --ply RESULTS/ply/point_cloud_29999.ply --align SCENE/align.json \
      --out SCENE/export --gsplat_examples ~/splat/src/gsplat/examples --calib calib/data/calib.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--ply", required=True)
ap.add_argument("--align", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gsplat_examples", required=True)
# desk footprint (desk frame, CCW): front edge x = -0.12, right edge and back edge from the overhead view
# (build_real2sim_assets.py desk_back_edge x = 0.376 - 0.061 y; right edge u = 470 px), left edge beyond the view
ap.add_argument("--desk_poly", type=float, nargs="+",
                default=(-0.12, 0.60, -0.12, -0.226, 0.386, -0.164, 0.339, 0.60), help="x0 y0 x1 y1 ... (m)")
ap.add_argument("--desk_z", type=float, nargs=2, default=(-0.30, 0.35), help="height band dropped over the desk")
ap.add_argument("--base_xy", type=float, nargs=2, default=(0.0, 0.0))
ap.add_argument("--crop_radius", type=float, default=0.42, help="workspace cylinder around the base (m)")
ap.add_argument("--crop_z", type=float, nargs=2, default=(-0.30, 0.35), help="height band dropped inside the cylinder")
ap.add_argument("--calib", default=None, help="calib.json: drop Gaussians near the overhead camera inside its view")
ap.add_argument("--cull_depth", type=float, default=0.35)
ap.add_argument("--extra_gain", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                help="RGB gain on top of SPLAT_COLOR_GAIN: the video was shot with the room lit, the episodes are not")
ap.add_argument("--texture", default="../real2sim/desk_albedo.png", help="desk texture, relative to OUT's USD")
ap.add_argument("--texture_quad", type=float, nargs=4, default=(-0.20, 0.54, -0.60, 0.60),
                help="x0 x1 y0 y1 the texture covers (backdrop.usda DeskTop quad)")
args = ap.parse_args()

sys.path.insert(0, args.gsplat_examples)
from datasets.colmap import Parser  # noqa: E402  (same normalisation as training)

T = Parser(args.scene, factor=1, normalize=True, test_every=8).transform  # 4x4: normalised = T @ colmap
A, b = T[:3, :3], T[:3, 3]
k = np.cbrt(np.linalg.det(A))
al = json.loads(Path(args.align).read_text())
s, R, t = al["SPLAT_SCALE"], Rotation.from_quat(al["SPLAT_QUAT"]).as_matrix(), np.asarray(al["SPLAT_POS"])
# desk = s R (A^-1 (x - b)) + t = (s / k) (R Q^T) x + (t - (s / k) R Q^T b), with A = k Q
Q = A / k
s2, R2 = s / k, R @ Q.T
t2 = t - s2 * R2 @ b

ply = PlyData.read(args.ply)
v = ply["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], -1) @ (s2 * R2).T + t2
poly = np.asarray(args.desk_poly, float).reshape(-1, 2)
desk = Polygon(poly).contains_points(xyz[:, :2]) & (xyz[:, 2] > args.desk_z[0]) & (xyz[:, 2] < args.desk_z[1])
r = np.linalg.norm(xyz[:, :2] - np.asarray(args.base_xy), axis=1)
work = (r < args.crop_radius) & (xyz[:, 2] > args.crop_z[0]) & (xyz[:, 2] < args.crop_z[1])
below = xyz[:, 2] < -1.0  # the floor is ~0.75 m below the desk top; nothing real is lower
mount = np.zeros(len(xyz), bool)
if args.calib:
    cal = json.loads(Path(args.calib).read_text())
    Kc = np.asarray(cal["SCENE_K"], float).reshape(3, 3)
    pc = (xyz - np.asarray(cal["SCENE_CAM_POS"])) @ Rotation.from_quat(cal["SCENE_CAM_QUAT"]).as_matrix()
    uv = pc[:, :2] / np.maximum(pc[:, 2:3], 1e-6) * [Kc[0, 0], Kc[1, 1]] + [Kc[0, 2], Kc[1, 2]]
    mount = (pc[:, 2] > -0.05) & (pc[:, 2] < args.cull_depth) & (uv[:, 0] > -80) & (uv[:, 0] < 720) & \
        (uv[:, 1] > -80) & (uv[:, 1] < 560)
keep = ~(desk | work | below | mount)
# colour gain (refine_splat_align.py: the overhead camera's exposure): rgb = 0.5 + C0 * f_dc, view-dependent terms scale too
g = np.asarray(al.get("SPLAT_COLOR_GAIN", (1.0, 1.0, 1.0)), float) * np.asarray(args.extra_gain)
data = v.data.copy()
C0 = 0.28209479177387814
for c in range(3):
    data[f"f_dc_{c}"] = (g[c] * (0.5 + C0 * data[f"f_dc_{c}"]) - 0.5) / C0
rest = sorted((n for n in data.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
for i, n in enumerate(rest):  # gsplat/INRIA layout: channel-major, (K coefficients) per channel
    data[n] *= g[i // (len(rest) // 3)]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
PlyData([PlyElement.describe(data[keep], "vertex")], text=False).write(str(out / "background.ply"))
pose = dict(SPLAT_POS=[float(x) for x in t2], SPLAT_QUAT=[float(x) for x in Rotation.from_matrix(R2).as_quat()],
            SPLAT_SCALE=float(s2), color_gain=[float(x) for x in g])
(out / "splat_pose.json").write_text(json.dumps(pose, indent=1))

# desk top for the sim: the backdrop's textured quad (same material layout, so the desk-texture randomization applies),
# clipped to the footprint; st follows the texture's original quad (s along -y, t along +x)
x0, x1, y0, y1 = args.texture_quad
pts = ", ".join(f"({x:.6f}, {y:.6f}, 0.000500)" for x, y in poly)
st = ", ".join(f"({(y1 - y) / (y1 - y0):.6f}, {(x - x0) / (x1 - x0):.6f})" for x, y in poly)
lo, hi = poly.min(0), poly.max(0)
(out / "desk_top.usda").write_text(f'''#usda 1.0
(
    defaultPrim = "Backdrop"
    doc = "Desk top for the splat background (visual only). Generated by calib/export_splat.py"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Backdrop"
{{
    def Mesh "DeskTop" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [({lo[0]:.6f}, {lo[1]:.6f}, 0.000500), ({hi[0]:.6f}, {hi[1]:.6f}, 0.000500)]
        int[] faceVertexCounts = [{len(poly)}]
        int[] faceVertexIndices = [{", ".join(str(i) for i in range(len(poly)))}]
        rel material:binding = </Backdrop/Looks/DeskMat>
        normal3f[] normals = [{", ".join(["(0, 0, 1)"] * len(poly))}] (
            interpolation = "vertex"
        )
        point3f[] points = [{pts}]
        texCoord2f[] primvars:st = [{st}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
    }}

    def Scope "Looks"
    {{
        def Material "DeskMat"
        {{
            token outputs:surface.connect = </Backdrop/Looks/DeskMat/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Backdrop/Looks/DeskMat/Texture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.6
                token outputs:surface
            }}

            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                float4 inputs:bias = (0, 0, 0, 0)
                asset inputs:file = @{args.texture}@
                float4 inputs:scale = (1, 1, 1, 1)
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </Backdrop/Looks/DeskMat/Reader.outputs:result>
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}

            def Shader "Reader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
''')
# IsaacLab spawns the referenced default prim as an Xform, which would override the ParticleField's type (nothing
# renders): reference the converted splat from a child of an Xform root instead
(out / "background_root.usda").write_text('''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root"
{
    def "splat" (
        prepend references = @./background.usdc@
    )
    {
    }
}
''')
print(f"[export] {len(xyz)} Gaussians: dropped {int(desk.sum())} over the desk, {int((work & ~desk).sum())} more in the "
      f"workspace, {int((mount & ~desk & ~work).sum())} on the overhead mount, {int(below.sum())} stray; kept "
      f"{int(keep.sum())} -> {out / 'background.ply'}; {out / 'desk_top.usda'}; pose {pose}")
