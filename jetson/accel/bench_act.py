#!/usr/bin/env python
"""GPU-only latency benchmark for a LeRobot ACT checkpoint on the Jetson (no robot, no cameras).

  python bench_act.py --ckpt ~/so101_policies/act_real_v0 --group eager|compile|cgcompile|prep
Prints one JSON line per measurement (prefix 'RES ').
"""
import argparse, contextlib, gc, inspect, json, os, pickle, socket, statistics, sys, textwrap, threading, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("~/so101_policies/act_real_v0"))
ap.add_argument("--group", default="eager")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()
PROFILE = args.profile

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies import prepare_observation_for_inference
import lerobot.policies.act.modeling_act as mact

GPU_FREQ = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"
dev = torch.device("cuda")


def emit(**kw):
    print("RES " + json.dumps(kw), flush=True)


def meminfo():
    rss = avail = None
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) / 1024
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / 1024
    return dict(rss_mb=round(rss), sys_avail_mb=round(avail),
                cuda_peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20),
                cuda_reserved_mb=round(torch.cuda.memory_reserved() / 2**20))


class FreqSampler:
    def __enter__(self):
        self.vals, self.stop = [], False
        def run():
            while not self.stop:
                try:
                    self.vals.append(int(open(GPU_FREQ).read()) // 1_000_000)
                except Exception:
                    pass
                time.sleep(0.02)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.stop = True; self.t.join()
    def median(self):
        return int(statistics.median(self.vals)) if self.vals else None


def timeit(fn, n=args.n, warmup=args.warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts, cs = [], []
    with FreqSampler() as fs:
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            cs.append((time.perf_counter() - t0) * 1000)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return dict(median_ms=round(statistics.median(ts), 2), p90_ms=round(ts[int(0.9 * (len(ts) - 1))], 2),
                min_ms=round(ts[0], 2), max_ms=round(ts[-1], 2), cpu_launch_ms=round(statistics.median(cs), 2),
                gpu_mhz=fs.median())


# ---------------------------------------------------------------- inputs: real frames from the user's dataset
def real_obs():
    root = Path.home() / "so101_datasets/cube_cup_real_v1"
    obs = []
    try:
        import av
        import pyarrow.parquet as pq
        tab = pq.read_table(root / "data/chunk-000/file-000.parquet", columns=["observation.state"])
        states = tab.column("observation.state").to_pylist()
        frames = {}
        for cam in ("wrist", "scene"):
            f = root / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
            with av.open(str(f)) as c:
                got = []
                for i, fr in enumerate(c.decode(video=0)):
                    if i in (30, 150, 300):
                        got.append(fr.to_ndarray(format="rgb24"))
                    if i > 300:
                        break
            frames[cam] = got
        for j, i in enumerate((30, 150, 300)):
            obs.append({"observation.state": np.asarray(states[i], np.float32),
                        "observation.images.wrist": frames["wrist"][j], "observation.images.scene": frames["scene"][j]})
    except Exception as e:  # fall back to noise
        emit(note=f"real frames unavailable: {type(e).__name__}: {e}")
        rng = np.random.default_rng(0)
        for _ in range(3):
            obs.append({"observation.state": rng.normal(0, 30, 6).astype(np.float32),
                        "observation.images.wrist": rng.integers(0, 255, (480, 640, 3), np.uint8),
                        "observation.images.scene": rng.integers(0, 255, (480, 640, 3), np.uint8)})
    return obs


def load():
    t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = "cuda"
    pol = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg).to(dev).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    nparam = sum(p.numel() for p in pol.parameters())
    emit(what="load", load_s=round(time.perf_counter() - t0, 2), params_M=round(nparam / 1e6, 2),
         torch=torch.__version__, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
         cap=torch.cuda.get_device_capability(), bf16=torch.cuda.is_bf16_supported(), **meminfo())
    return cfg, pol, pre, post


def prep(pre, o):
    return pre(prepare_observation_for_inference(dict(o), dev, "Put the cube in the cup", "so_follower"))


def patch_act_for_graphs():
    """ACT.forward builds the zero latent on the CPU then .to(device): a host->device copy that is illegal while a
    CUDA graph is being captured. Rebuild it directly on the device (numerically identical)."""
    import re
    src = textwrap.dedent(inspect.getsource(mact.ACT.forward))
    pat = r"torch\.zeros\(\[batch_size, self\.config\.latent_dim\], dtype=torch\.float32\)\.to\(\s*batch\[OBS_STATE\]\.device\s*\)"
    src, n = re.subn(pat, "torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=batch[OBS_STATE].device)", src)
    if n != 1:
        raise RuntimeError("ACT.forward source changed; cannot patch the CPU latent")
    ns = {}
    exec(compile(src, mact.__file__, "exec"), mact.__dict__, ns)
    mact.ACT.forward = ns["forward"]


class Graphed:
    """Capture policy.predict_action_chunk once in a CUDA graph; replay with copied-in inputs."""

    def __init__(self, pol, batch, amp_dtype=None, warm=3):
        self.keys = [k for k, v in batch.items() if torch.is_tensor(v)]
        self.static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.pol = pol
        self.amp = (lambda: torch.autocast("cuda", dtype=amp_dtype, cache_enabled=False)) if amp_dtype else contextlib.nullcontext
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode(), self.amp():
            for _ in range(warm):
                pol.predict_action_chunk(self.static)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), self.amp(), torch.cuda.graph(self.g):
            self.out = pol.predict_action_chunk(self.static)

    def __call__(self, batch):
        with torch.inference_mode():
            for k in self.keys:
                self.static[k].copy_(batch[k])
            self.g.replay()
            return self.out.clone()


def kernel_ms(fn):
    try:
        from torch.profiler import profile, ProfilerActivity
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        tot = 0.0
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                tot += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        return round(tot / 1000, 2)
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:120]


def model_variant(name, pol, batches, amp_dtype=None, fn=None, ref=None, post=None):
    torch.cuda.reset_peak_memory_stats()
    amp = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
    if fn is None:
        def fn(b):
            with torch.inference_mode(), amp():
                return pol.predict_action_chunk(b)
    b0 = batches[0]
    r = timeit(lambda: fn(b0))
    # accuracy vs fp32 reference, in the robot's units (degrees / gripper 0-100) after unnormalization
    diffs = []
    outs = []
    with torch.inference_mode():
        for b in batches:
            a = post(fn(b).float()).cpu().numpy()[0]
            outs.append(a)
    if ref is not None:
        diffs = [float(np.abs(a - rr).max()) for a, rr in zip(outs, ref)]
    km = kernel_ms(lambda: fn(b0)) if PROFILE else None
    emit(what="model", variant=name, **r, gpu_kernel_sum_ms=km, max_abs_diff_deg=(round(max(diffs), 4) if diffs else None), **meminfo())
    return outs


def group_eager():
    cfg, pol, pre, post = load()
    obs = real_obs()
    with torch.inference_mode():
        batches = [prep(pre, o) for o in obs]
    torch.backends.cudnn.benchmark = False
    ref = model_variant("fp32 (worker default: cudnn.benchmark off, TF32 matmul off)", pol, batches, post=post)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model_variant("fp32 + cudnn.benchmark + TF32 (lerobot_eval defaults)", pol, batches, ref=ref, post=post)
    model_variant("autocast fp16", pol, batches, torch.float16, ref=ref, post=post)
    model_variant("autocast bf16", pol, batches, torch.bfloat16, ref=ref, post=post)
    pol.model.backbone.to(memory_format=torch.channels_last)
    batches_cl = []
    for b in batches:
        b2 = dict(b)
        for k in ("observation.images.wrist", "observation.images.scene"):
            b2[k] = b[k].contiguous(memory_format=torch.channels_last)
        batches_cl.append(b2)
    model_variant("autocast fp16 + channels_last backbone", pol, batches_cl, torch.float16, ref=ref, post=post)
    # CUDA graph (needs the 1-line latent patch)
    patch_act_for_graphs()
    try:
        g = Graphed(pol, batches_cl[0], torch.float16)
        model_variant("autocast fp16 + channels_last + CUDA graph", pol, batches_cl, fn=g, ref=ref, post=post)
        del g
    except Exception as e:
        emit(what="model", variant="fp16 + CUDA graph", error=f"{type(e).__name__}: {e}"[:400])
    pol.model.backbone.to(memory_format=torch.contiguous_format)
    try:
        g = Graphed(pol, batches[0], None)
        model_variant("fp32 + TF32 + CUDA graph", pol, batches, fn=g, ref=ref, post=post)
        del g
    except Exception as e:
        emit(what="model", variant="fp32 + CUDA graph", error=f"{type(e).__name__}: {e}"[:400])
    # pure fp16 weights (still under autocast so the fp32 latent/state/pos-embeds are cast per op)
    pol.half()
    pol.model.backbone.to(memory_format=torch.channels_last)
    gc.collect(); torch.cuda.empty_cache()
    model_variant("fp16 weights (.half()) + autocast + channels_last", pol, batches_cl, torch.float16, ref=ref, post=post)
    try:
        g = Graphed(pol, batches_cl[0], torch.float16)
        model_variant("fp16 weights + channels_last + CUDA graph", pol, batches_cl, fn=g, ref=ref, post=post)
        del g
    except Exception as e:
        emit(what="model", variant="fp16 weights + CUDA graph", error=f"{type(e).__name__}: {e}"[:400])


def group_compile(backend, mode):
    cfg, pol, pre, post = load()
    obs = real_obs()
    with torch.inference_mode():
        batches = [prep(pre, o) for o in obs]
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    ref = model_variant("fp32 ref (this process)", pol, batches, post=post)
    if backend == "cudagraphs" or mode == "reduce-overhead":
        patch_act_for_graphs()
    if os.environ.get("FREEZE") == "1":
        import torch._inductor.config as icfg
        icfg.freezing = True
    pol.model.backbone.to(memory_format=torch.channels_last)
    batches_cl = []
    for b in batches:
        b2 = dict(b)
        for k in ("observation.images.wrist", "observation.images.scene"):
            b2[k] = b[k].contiguous(memory_format=torch.channels_last)
        batches_cl.append(b2)
    kw = dict(backend=backend) if mode is None else dict(backend=backend, mode=mode)
    compiled = torch.compile(pol.predict_action_chunk, **kw)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            compiled(batches_cl[0]); compiled(batches_cl[0]); torch.cuda.synchronize()
    except Exception as e:
        emit(what="model", variant=f"torch.compile {backend}/{mode} fp16+CL", compile_s=round(time.perf_counter() - t0, 1),
             error=f"{type(e).__name__}: {e}"[-1500:])
        return
    cs = round(time.perf_counter() - t0, 1)
    emit(what="compile_done", backend=backend, mode=mode, compile_s=cs)

    def fn(b):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return compiled(b)
    model_variant(f"torch.compile {backend}/{mode} freezing={os.environ.get('FREEZE') == '1'} + autocast fp16 + CL (compile {cs}s)", pol, batches_cl, fn=fn, ref=ref, post=post)
    o = obs[0]
    def e2e():
        with torch.inference_mode():
            bb = to_cl([prep(pre, o)])[0]
            return post(fn(bb).float()).cpu().numpy()
    emit(what="e2e", variant=f"compiled {mode} full predict (prep+model+post)", **timeit(e2e), **meminfo())
    for gap in (0.33, 1.0):
        ts, fr = [], []
        for i in range(12):
            time.sleep(gap)
            fr.append(int(open(GPU_FREQ).read()) // 1_000_000)
            t0 = time.perf_counter(); e2e(); ts.append((time.perf_counter() - t0) * 1000)
        ts, fr = ts[3:], fr[3:]
        emit(what="duty", variant=f"compiled {mode} full predict, idle gap {gap}s", median_ms=round(statistics.median(ts), 2),
             max_ms=round(max(ts), 2), gpu_mhz_before_call=sorted(set(fr)))


def group_prep():
    import cv2
    cfg, pol, pre, post = load()
    obs = real_obs()
    o = obs[0]
    torch.backends.cudnn.benchmark = True

    def step_prepare():
        with torch.inference_mode():
            return prepare_observation_for_inference(dict(o), dev, "t", "so_follower")
    emit(what="prep", step="prepare_observation_for_inference (2x uint8 480x640 -> cuda float /255 CHW + state)", **timeit(step_prepare))
    with torch.inference_mode():
        p = prepare_observation_for_inference(dict(o), dev, "t", "so_follower")

    def step_pre():
        with torch.inference_mode():
            return pre(dict(p))
    emit(what="prep", step="preprocessor pipeline (rename, batch, device, MEAN_STD normalize)", **timeit(step_pre))

    def step_all_pre():
        with torch.inference_mode():
            return prep(pre, o)
    emit(what="prep", step="prepare + preprocessor (total pre)", **timeit(step_all_pre))
    with torch.inference_mode():
        b = prep(pre, o)
        a = pol.predict_action_chunk(b)

    def step_post():
        with torch.inference_mode():
            return post(a).float().cpu().numpy()
    emit(what="prep", step="postprocessor (unnormalize) + .cpu().numpy()", **timeit(step_post))

    img = o["observation.images.scene"]
    emit(what="prep", step="cv2.resize 640x480 -> 320x240 INTER_AREA (1 image, CPU)",
         **timeit(lambda: cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)))
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    emit(what="prep", step="cv2.imencode JPEG q90 640x480 (1 image, CPU)",
         **timeit(lambda: cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])), jpeg_kb=round(len(jpg) / 1024, 1))
    emit(what="prep", step="cv2.imdecode JPEG + BGR2RGB 640x480 (1 image, CPU)",
         **timeit(lambda: cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
    # IPC: pickled dict with two raw frames over a unix socketpair (what policy_client does locally)
    from multiprocessing.connection import Pipe
    a_end, b_end = Pipe(duplex=True)
    msg = {"cmd": "chunk", "id": 1, "task": "t", "obs": dict(o)}
    def echo():
        while True:
            m = b_end.recv()
            if m is None:
                return
            b_end.send({"id": m["id"], "actions": np.zeros((50, 6), np.float32)})
    th = threading.Thread(target=echo, daemon=True); th.start()
    def rt():
        a_end.send(msg); a_end.recv()
    emit(what="prep", step="IPC round trip: pickle 2 raw frames over unix socketpair + reply", **timeit(rt))
    a_end.send(None)
    # mimic policy_worker.predict end to end, fp32 vs fp16 autocast
    for name, amp in (("fp32", None), ("autocast fp16", torch.float16)):
        ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp else contextlib.nullcontext
        def e2e():
            with torch.inference_mode(), ctx():
                bb = prep(pre, o)
                aa = pol.predict_action_chunk(bb)
                return post(aa).float().cpu().numpy()
        emit(what="e2e", variant=f"policy_worker.predict equivalent, {name}", **timeit(e2e), **meminfo())
    pol.model.backbone.to(memory_format=torch.channels_last)
    patch_act_for_graphs()
    with torch.inference_mode():
        b0 = to_cl([prep(pre, o)])[0]
    g = Graphed(pol, b0, torch.float16)
    def e2e_graph():
        with torch.inference_mode():
            bb = prep(pre, o)
            aa = g(bb)
            return post(aa.float()).cpu().numpy()
    emit(what="e2e", variant="prepare + preprocessor + CUDA-graph fp16/CL + postprocessor + .cpu()", **timeit(e2e_graph), **meminfo())


def to_cl(batches):
    out = []
    for b in batches:
        b2 = dict(b)
        for k in ("observation.images.wrist", "observation.images.scene"):
            b2[k] = b[k].contiguous(memory_format=torch.channels_last)
        out.append(b2)
    return out


def group_parts():
    """Where does the GPU time go? CUDA-graph each part in isolation (fp16 autocast, channels_last)."""
    cfg, pol, pre, post = load()
    obs = real_obs()
    with torch.inference_mode():
        batches = to_cl([prep(pre, o) for o in obs])
    torch.backends.cudnn.benchmark = True
    pol.model.backbone.to(memory_format=torch.channels_last)
    patch_act_for_graphs()
    img = batches[0]["observation.images.wrist"]
    img2 = torch.cat([batches[0]["observation.images.wrist"], batches[0]["observation.images.scene"]]).contiguous(
        memory_format=torch.channels_last)

    def graph_of(f, x):
        static = x.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            for _ in range(3):
                f(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False), torch.cuda.graph(g):
            out = f(static)
        return g, out
    bb = lambda x: pol.model.backbone(x)["feature_map"]
    g1, _ = graph_of(bb, img)
    emit(what="part", part="ResNet18 backbone, 1 image 480x640 (x2 per inference)", **timeit(g1.replay))
    g2, _ = graph_of(bb, img2)
    emit(what="part", part="ResNet18 backbone, both images batched (B=2)", **timeit(g2.replay))
    gfull = Graphed(pol, batches[0], torch.float16)
    emit(what="part", part="full predict_action_chunk (graph)", **timeit(lambda: gfull(batches[0])))
    # duty cycle: one inference per second like the real control loop (n_action_steps 30 @ 30 fps)
    for name, fn in (("fp16+CL CUDA graph", lambda: gfull(batches[0])),):
        ts, fr = [], []
        for i in range(12):
            time.sleep(1.0)
            fr.append(int(open(GPU_FREQ).read()) // 1_000_000)
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        ts = ts[2:]
        emit(what="duty", variant=f"{name}, 1 call per 1.0 s idle gap", median_ms=round(statistics.median(ts), 2),
             max_ms=round(max(ts), 2), min_ms=round(min(ts), 2), gpu_mhz_before_call=fr[2:])
    def eager():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return pol.predict_action_chunk(batches[0])
    ts, fr = [], []
    for i in range(12):
        time.sleep(1.0)
        fr.append(int(open(GPU_FREQ).read()) // 1_000_000)
        t0 = time.perf_counter(); eager(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    ts = ts[2:]
    emit(what="duty", variant="fp16+CL eager, 1 call per 1.0 s idle gap", median_ms=round(statistics.median(ts), 2),
         max_ms=round(max(ts), 2), min_ms=round(min(ts), 2), gpu_mhz_before_call=fr[2:])


def fold_frozen_bn(backbone):
    """Fold torchvision FrozenBatchNorm2d into the preceding conv (ResNet conv->bn pairs). Same math, fewer kernels."""
    import torch.nn as nn
    from torchvision.ops.misc import FrozenBatchNorm2d

    def fold(conv, bn):
        scale = bn.weight * (bn.running_var + bn.eps).rsqrt()
        bias = bn.bias - bn.running_mean * scale
        conv.weight.data.mul_(scale.view(-1, 1, 1, 1))
        if conv.bias is None:
            conv.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            conv.bias.data = conv.bias.data * scale + bias
    n = 0
    for _, m in list(backbone.named_modules()):
        for c, b in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
            if isinstance(getattr(m, c, None), nn.Conv2d) and isinstance(getattr(m, b, None), FrozenBatchNorm2d):
                fold(getattr(m, c), getattr(m, b)); setattr(m, b, nn.Identity()); n += 1
        if isinstance(m, nn.Sequential) and len(m) == 2 and isinstance(m[0], nn.Conv2d) and isinstance(m[1], FrozenBatchNorm2d):
            fold(m[0], m[1]); m[1] = nn.Identity(); n += 1
    return n


def group_final(cudnn_bench, fold):
    """Fresh process: the recommended config end to end, plus duty-cycle latency at several replan cadences."""
    torch.backends.cudnn.benchmark = cudnn_bench
    cfg, pol, pre, post = load()
    obs = real_obs()
    with torch.inference_mode():
        batches = [prep(pre, o) for o in obs]
    torch.cuda.reset_peak_memory_stats()
    ref = []
    with torch.inference_mode():
        for b in batches:
            ref.append(post(pol.predict_action_chunk(b)).float().cpu().numpy()[0])
    nf = fold_frozen_bn(pol.model.backbone) if fold else 0
    pol.model.backbone.to(memory_format=torch.channels_last)
    patch_act_for_graphs()
    t0 = time.perf_counter()
    g = Graphed(pol, to_cl(batches)[0], torch.float16)
    cap_s = time.perf_counter() - t0
    tag = f"fp16+CL+CUDA graph, cudnn.benchmark={cudnn_bench}, bn_fold={fold}({nf})"
    model_variant(tag + f" [capture {cap_s:.1f}s]", pol, to_cl(batches), fn=g, ref=ref, post=post)
    o = obs[0]
    def e2e():
        with torch.inference_mode():
            return post(g(prep(pre, o)).float()).cpu().numpy()
    emit(what="e2e", variant=tag + " full predict (prep+graph+post)", **timeit(e2e), **meminfo())
    for gap in (0.1, 0.33, 1.0):
        ts, fr = [], []
        for i in range(14):
            time.sleep(gap)
            fr.append(int(open(GPU_FREQ).read()) // 1_000_000)
            t0 = time.perf_counter(); e2e(); ts.append((time.perf_counter() - t0) * 1000)
        ts, fr = ts[4:], fr[4:]
        emit(what="duty", variant=tag + f" full predict, idle gap {gap}s", median_ms=round(statistics.median(ts), 2),
             max_ms=round(max(ts), 2), gpu_mhz_before_call=sorted(set(fr)))


if __name__ == "__main__":
    emit(what="env", group=args.group, gpu_freq_now_mhz=int(open(GPU_FREQ).read()) // 1_000_000,
         power_mode=os.popen("nvpmodel -q 2>/dev/null | head -1").read().strip())
    if args.group == "eager":
        group_eager()
    elif args.group.startswith("final"):
        # final_<bench 0/1>_<fold 0/1>
        _, b, f = args.group.split("_")
        group_final(b == "1", f == "1")
    elif args.group == "parts":
        group_parts()
    elif args.group == "prep":
        group_prep()
    elif args.group == "compile":
        group_compile("inductor", "default")
    elif args.group == "compile_maxau":
        group_compile("inductor", "max-autotune-no-cudagraphs")
    elif args.group == "compile_ro":
        group_compile("inductor", "reduce-overhead")
    elif args.group == "cgcompile":
        group_compile("cudagraphs", None)
    emit(what="done", group=args.group)
