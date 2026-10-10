"""Check jetson_accel.install() on a real checkpoint: outputs vs fp32 and latency through the policy_worker path."""
import os, statistics, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_accel
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies import prepare_observation_for_inference

ckpt = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/so101_policies/act_real_v0")
MODE = sys.argv[2] if len(sys.argv) > 2 else "auto"
cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.device = "cuda"
pol = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": "cuda"}})
import av, pyarrow.parquet as pq
from pathlib import Path
root = Path.home() / "so101_datasets/cube_cup_real_v1"
states = pq.read_table(root / "data/chunk-000/file-000.parquet", columns=["observation.state"]).column("observation.state").to_pylist()
frames = {}
for cam in ("wrist", "scene"):
    with av.open(str(root / f"videos/observation.images.{cam}/chunk-000/file-000.mp4")) as c:
        frames[cam] = [f.to_ndarray(format="rgb24") for i, f in zip(range(301), c.decode(video=0)) if i in (30, 100, 200, 300)]
obs = [{"observation.state": np.asarray(states[i], np.float32), "observation.images.wrist": frames["wrist"][j],
        "observation.images.scene": frames["scene"][j]} for j, i in enumerate((30, 100, 200, 300))]

def predict(o):  # == policy_worker.Policy.predict (minus amp, which install() handles)
    with torch.inference_mode():
        b = pre(prepare_observation_for_inference(dict(o), torch.device("cuda"), "Put the cube in the cup", "so_follower"))
        a = pol.predict_action_chunk(b)
        return post(a).float().cpu().numpy()[0]

ref = [predict(o) for o in obs]
t0 = time.perf_counter(); info = jetson_accel.install(pol, mode=MODE); print("install", info, f"{time.perf_counter()-t0:.2f}s")
t0 = time.perf_counter(); predict(obs[0]); print(f"first call (capture/compile) {1000*(time.perf_counter()-t0):.0f} ms")
outs = [predict(o) for o in obs]
print("max |diff| per frame (deg / gripper units):", [round(float(np.abs(a - r).max()), 4) for a, r in zip(outs, ref)])
# chunks must be independent objects (graph buffer not leaked)
a1 = predict(obs[0]); a2 = predict(obs[1]); print("independent outputs:", not np.allclose(a1, a2), round(float(np.abs(a1 - ref[0]).max()), 4))
ts = []
for i in range(40):
    t0 = time.perf_counter(); predict(obs[i % 4]); ts.append(1000 * (time.perf_counter() - t0))
ts = sorted(ts[10:]); print(f"full predict median {statistics.median(ts):.2f} ms p90 {ts[int(0.9*(len(ts)-1))]:.2f} ms")
# select_action path (lerobot-rollout style) still works
pol.reset()
with torch.inference_mode():
    b = pre(prepare_observation_for_inference(dict(obs[0]), torch.device("cuda"), "x", "so_follower"))
    acts = [pol.select_action(b) for _ in range(31)]
print("select_action ok:", tuple(acts[0].shape), "queue refilled after", cfg.n_action_steps, "steps")
print("cuda peak alloc MB", round(torch.cuda.max_memory_allocated() / 2**20), "reserved MB", round(torch.cuda.memory_reserved() / 2**20))
