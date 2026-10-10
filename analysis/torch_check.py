"""Run the real mdp.reset_cube_and_cup on CPU with stubbed Isaac modules; check desk constraint + fallback path."""
import sys, types, importlib.util
import numpy as np, torch
for m in ("isaaclab", "isaaclab.managers", "isaaclab.utils", "isaaclab.envs"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["isaaclab.managers"].SceneEntityCfg = object
mu = types.ModuleType("isaaclab.utils.math")
mu.quat_from_euler_xyz = lambda r, p, y: torch.stack([torch.zeros_like(y)] * 3 + [torch.ones_like(y)], -1)
sys.modules["isaaclab.utils"].math = mu
sys.modules["isaaclab.utils.math"] = mu
spec = importlib.util.spec_from_file_location("mdp", "/home/asubuntudesktop/TestingGrounds/so101_r2s2r/so101_r2s2r/mdp.py")
mdp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdp)
sys.path.insert(0, "."); import sampler_check as sc  # noqa (runs its report on import with N from argv)

class Obj:
    def __init__(s): s.pose = None
    def write_root_pose_to_sim_index(s, root_pose, env_ids): s.pose = root_pose.clone()
    def write_root_velocity_to_sim_index(s, root_velocity, env_ids): pass
class Scene(dict):
    pass
class Cfg: cube_sizes = (0.02, 0.03, 0.04); default_cube = 1; cup_dims = dict(top_d=0.080)
class Env:
    def __init__(s, n):
        s.num_envs, s.device, s.cfg = n, "cpu", Cfg()
        s.scene = Scene(cup=Obj(), cube_s=Obj(), cube_m=Obj(), cube_l=Obj()); s.scene.env_origins = torch.zeros(n, 3)

P = dict(cube_rphi={"r": (0.12, 0.27), "phi": (-1.1, 1.1)}, cup_rphi={"r": (0.15, 0.32), "phi": (-1.2, 1.2)},
         min_separation=0.11, cup_keepout=((0.17, 0.0), 0.10), desk_poly=sc.DESK_POLY.tolist(), desk_margin=0.015,
         num_candidates=64)
torch.manual_seed(0)
for probs in ((0.0, 1.0, 0.0), (1/3, 1/3, 1/3)):
    n = 200_000; env = Env(n)
    mdp.reset_cube_and_cup(env, torch.arange(n), cube_probs=probs, **P)
    cup = env.scene["cup"].pose[:, :2].numpy()
    idx = env.r2s_active_cube.numpy(); size = np.array(Cfg.cube_sizes)[idx]
    cube = np.stack([env.scene[nm].pose[:, :2].numpy() for nm in mdp.CUBE_NAMES], 1)[np.arange(n), idx]
    dc, du = sc.desk_inside(cube), sc.desk_inside(cup)
    sep = np.linalg.norm(cube - cup, axis=-1); keep = np.linalg.norm(cup - [0.17, 0.0], axis=-1)
    print(f"probs {probs}: fallbacks {env.r2s_spawn_stats}; cube min inside-margin {np.min(dc - size/2)*100:.2f} cm "
          f"(>=1.5); cup min inside {du.min()*100:.2f} cm (>=5.5); min sep {sep.min():.3f} (>0.11); min keepout {keep.min():.3f} (>0.10)")
    # torch signed distance == numpy replica
    t = mdp._inside_convex(torch.tensor(cube[:1000]), sc.DESK_POLY.tolist()).numpy()
    assert np.allclose(t, sc.desk_inside(cube[:1000]), atol=1e-6)
    t2 = mdp._inside_convex(torch.tensor(cube[:1000]), sc.DESK_POLY[::-1].tolist()).numpy()  # CW winding
    assert np.allclose(t2, t, atol=1e-6)
# fallback path: impossible cup range (all off the desk) -> every env falls back, cube still on desk
env = Env(8)
Q = dict(P, cup_rphi={"r": (0.40, 0.40), "phi": (-1.4, -1.4)})
mdp.reset_cube_and_cup(env, torch.arange(8), cube_probs=(0.0, 1.0, 0.0), **Q)
print("fallback stats", env.r2s_spawn_stats, "cup", env.scene["cup"].pose[0, :2].numpy())
# desk_poly=None -> old behaviour (fallback = first candidate), no counting noise
env = Env(1000); mdp.reset_cube_and_cup(env, torch.arange(1000), cube_probs=(0.0, 1.0, 0.0), **dict(P, desk_poly=None))
cup = env.scene["cup"].pose[:, :2].numpy(); print("desk_poly=None: cup off desk", np.mean(sc.desk_inside(cup) < 0), env.r2s_spawn_stats)
