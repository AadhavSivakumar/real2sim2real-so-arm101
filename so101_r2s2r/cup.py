"""Procedural tapered cup as a USD rigid body.

Visual: a smooth open tapered cylinder (outer + inner wall, rim, floor); two materials like the real cup
(white outside + rim, light blue inside), bound to the GeomSubsets "outer" / "inner".
Collision: PhysX dynamic bodies need convex shapes, so the hollow is built from thin box "staves"
around the wall plus a cylinder floor -- the cube can actually drop inside.

Run standalone with Isaac Sim's python (needs pxr):
    ./lab.sh -p so101_r2s2r/cup.py <out.usda> [top_d bottom_d height] (metres)
"""
from __future__ import annotations

import math
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

# Placeholder until the real cup is measured (outer diameters / height in metres). Colours: linear albedo matched
# to the real overhead frames (scripts/compare_real_sim.py).
DEFAULT_CUP = dict(top_d=0.080, bottom_d=0.060, height=0.095, wall=0.002, floor=0.003, mass=0.03,
                   color=(0.80, 0.80, 0.78), inner_color=(0.55, 0.78, 0.86))


def _ring(r: float, z: float, n: int) -> list[Gf.Vec3f]:
    return [Gf.Vec3f(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def _visual_mesh(stage: Usd.Stage, path: str, top_r: float, bot_r: float, h: float, wall: float, floor: float, n: int = 48):
    """Watertight cup shell: outer wall, rim, inner wall, inner floor, outer floor. Returns (mesh, outer face
    indices, inner face indices)."""
    rings = [
        _ring(bot_r, 0.0, n),  # 0 outer bottom
        _ring(top_r, h, n),  # 1 outer top
        _ring(top_r - wall, h, n),  # 2 inner top
        _ring(bot_r - wall, floor, n),  # 3 inner bottom
    ]
    pts = [p for ring in rings for p in ring] + [Gf.Vec3f(0, 0, floor), Gf.Vec3f(0, 0, 0)]
    c_in, c_out = 4 * n, 4 * n + 1
    counts, idx = [], []
    for a, b in [(0, 1), (1, 2), (2, 3)]:  # outer wall, rim, inner wall (quads between consecutive rings)
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            idx += [a * n + i, a * n + j, b * n + j, b * n + i]
    for i in range(n):  # inner floor (fan, facing up) and outer floor (fan, facing down)
        j = (i + 1) % n
        counts.append(3); idx += [c_in, 3 * n + j, 3 * n + i]
        counts.append(3); idx += [c_out, 0 * n + i, 0 * n + j]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    # face order: n outer-wall quads, n rim quads, n inner-wall quads, then (inner floor, outer floor) per segment
    outer = list(range(2 * n)) + [3 * n + 2 * i + 1 for i in range(n)]
    inner = list(range(2 * n, 3 * n)) + [3 * n + 2 * i for i in range(n)]
    return mesh, outer, inner


def _preview_material(stage: Usd.Stage, path: str, color, roughness: float) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def make_cup_usd(out_path: str, top_d: float, bottom_d: float, height: float, wall: float = 0.002,
                 floor: float = 0.003, mass: float = 0.03, color=(0.80, 0.80, 0.78), inner_color=(0.55, 0.78, 0.86),
                 n_staves: int = 20) -> str:
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Cup")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)

    top_r, bot_r = top_d / 2, bottom_d / 2
    vis, outer, inner = _visual_mesh(stage, "/Cup/visual", top_r, bot_r, height, wall, floor)
    mats = {name: _preview_material(stage, f"/Cup/Looks/cup_{name}", c, 0.45) for name, c in (("outer", color), ("inner", inner_color))}
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(mats["outer"])
    for name, faces in (("outer", outer), ("inner", inner)):
        sub = UsdGeom.Subset.CreateGeomSubset(vis, name, UsdGeom.Tokens.face, Vt.IntArray(faces), UsdShade.Tokens.materialBind)
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mats[name])

    # Collision staves: thin boxes tangent to the (tapered) wall, tilted by the taper angle.
    cols = UsdGeom.Scope.Define(stage, "/Cup/collisions")
    taper = math.atan2(top_r - bot_r, height)  # outward lean of the wall
    slant = math.hypot(height, top_r - bot_r)
    r_mid = (top_r + bot_r) / 2 - wall / 2
    width = 2 * math.pi * ((top_r + bot_r) / 2) / n_staves * 1.15  # slight overlap closes the gaps
    thick = max(wall, 0.003)
    for k in range(n_staves):
        th = 2 * math.pi * k / n_staves
        box = UsdGeom.Cube.Define(stage, f"/Cup/collisions/stave_{k:02d}")
        box.CreateSizeAttr(1.0)
        box.CreatePurposeAttr(UsdGeom.Tokens.guide)  # invisible to renders
        xf = UsdGeom.Xformable(box)
        xf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(th), r_mid * math.sin(th), height / 2))
        # Gf composes left-to-right: lean outward about local y first, then spin to angle th about z.
        q = (Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(taper)) * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(th))).GetQuat()
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(q))
        xf.AddScaleOp().Set(Gf.Vec3f(thick, width, slant))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    base = UsdGeom.Cylinder.Define(stage, "/Cup/collisions/floor")
    base.CreateRadiusAttr(bot_r); base.CreateHeightAttr(floor); base.CreateAxisAttr("Z")
    base.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3d(0, 0, floor / 2))
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().customLayerData = {"cup": dict(top_d=top_d, bottom_d=bottom_d, height=height, wall=wall)}
    stage.Save()
    return out_path


if __name__ == "__main__":
    out = sys.argv[1]
    dims = dict(DEFAULT_CUP)
    if len(sys.argv) >= 5:
        dims.update(top_d=float(sys.argv[2]), bottom_d=float(sys.argv[3]), height=float(sys.argv[4]))
    print("wrote", make_cup_usd(out, **dims))
