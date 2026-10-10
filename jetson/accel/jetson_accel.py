"""Fast LeRobot ACT inference on a Jetson (Orin Nano Super, torch 2.11 cu128): no new packages needed.

Measured on act_real_v0 (2 cams 480x640, chunk 50), 25W mode, GPU 918 MHz, back-to-back calls, model only:
                                                                   model only   full predict (prep+model+post)
  fp32 eager (today's policy_worker)                                69.9 ms      73.5 ms
  mode="graph":   fp16 + channels_last + folded BN + CUDA graph     37.2 ms      40.2 ms  (first call +0.5 s)
  mode="compile": torch.compile reduce-overhead + freezing + fp16   19.4-21.7 ms 23.6-24.0 ms (first call 261 s cold,
                  43 s with a warm TORCHINDUCTOR_CACHE_DIR; needs Python.h: `sudo apt install python3.12-dev`)
  max |action diff| vs fp32 <= 0.06 (degrees for joints, 0-100 units for the gripper).
  At one inference per second the GPU idles down to 306 MHz between calls and every mode is ~18-20 ms slower;
  `sudo jetson_clocks` removes that.

Use:
    import jetson_accel
    jetson_accel.install(policy, mode="auto")   # after policy.to("cuda").eval(); ACT only (fixed input shapes)
    ... policy.predict_action_chunk(batch) / policy.select_action(batch) as usual; the first call compiles/captures

What it does (each step is optional, see install()):
  1. folds torchvision FrozenBatchNorm2d into the preceding conv of the ResNet backbone (same math, 20 fewer kernels)
  2. channels_last backbone (tensor-core friendly NHWC convs)
  3. runs predict_action_chunk under torch.autocast(fp16)
  4. captures the whole predict_action_chunk in one CUDA graph: the A78 CPU needs ~50 ms just to launch ACT's
     kernels plus the per-token Python work (eager fp16 is CPU-bound), a graph replay takes ~0.5 ms of CPU
For (4) it patches one line of ACT.forward that builds the zero VAE latent on the CPU and copies it to the GPU
(illegal during graph capture); the patched line creates it on the GPU directly (numerically identical).
"""
import contextlib
import inspect
import logging
import re
import textwrap

import torch

log = logging.getLogger("jetson_accel")
_PATCHED = False


def fold_frozen_bn(backbone):
    """Fold FrozenBatchNorm2d into the preceding Conv2d (ResNet conv1/bn1, block convN/bnN, downsample[0]/[1])."""
    import torch.nn as nn
    from torchvision.ops.misc import FrozenBatchNorm2d

    def fold(conv, bn):
        scale = bn.weight * (bn.running_var + bn.eps).rsqrt()
        bias = bn.bias - bn.running_mean * scale
        conv.weight.data.mul_(scale.view(-1, 1, 1, 1).to(conv.weight.dtype))
        if conv.bias is None:
            conv.bias = nn.Parameter(bias.detach().clone().to(conv.weight.dtype), requires_grad=False)
        else:
            conv.bias.data = conv.bias.data * scale + bias

    n = 0
    with torch.no_grad():
        for _, m in list(backbone.named_modules()):
            for c, b in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
                if isinstance(getattr(m, c, None), nn.Conv2d) and isinstance(getattr(m, b, None), FrozenBatchNorm2d):
                    fold(getattr(m, c), getattr(m, b))
                    setattr(m, b, nn.Identity())
                    n += 1
            if (isinstance(m, nn.Sequential) and len(m) == 2 and isinstance(m[0], nn.Conv2d)
                    and isinstance(m[1], FrozenBatchNorm2d)):
                fold(m[0], m[1])
                m[1] = nn.Identity()
                n += 1
    return n


def patch_act_latent():
    """Make ACT.forward create its zero latent on the GPU (needed for CUDA-graph capture). Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    import lerobot.policies.act.modeling_act as mact
    src = textwrap.dedent(inspect.getsource(mact.ACT.forward))
    pat = (r"torch\.zeros\(\[batch_size, self\.config\.latent_dim\], dtype=torch\.float32\)"
           r"\.to\(\s*batch\[OBS_STATE\]\.device\s*\)")
    src, n = re.subn(pat, "torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, "
                          "device=batch[OBS_STATE].device)", src)
    if n != 1:
        raise RuntimeError("lerobot ACT.forward changed: cannot patch the CPU-built latent (CUDA graph unsafe)")
    ns = {}
    exec(compile(src, mact.__file__, "exec"), mact.__dict__, ns)
    mact.ACT.forward = ns["forward"]
    _PATCHED = True


class GraphedChunk:
    """predict_action_chunk captured once in a CUDA graph; replays copy new inputs into static buffers."""

    def __init__(self, fn, batch, amp_dtype=torch.float16, warmup=3):
        self.fn = fn
        self.amp_dtype = amp_dtype
        self.keys = [k for k, v in batch.items() if torch.is_tensor(v)]
        self.sig = self.signature(batch)
        with torch.inference_mode():
            self.static = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    fmt = torch.channels_last if v.dim() == 4 else torch.contiguous_format
                    self.static[k] = v.clone(memory_format=fmt)
                else:
                    self.static[k] = v
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode(), self._amp():
            for _ in range(warmup):  # cuDNN algo selection + allocator warm-up outside the capture
                fn(self.static)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), self._amp(), torch.cuda.graph(self.graph):
            self.out = fn(self.static)

    def _amp(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype, cache_enabled=False)  # cache must be off in capture

    @staticmethod
    def signature(batch):
        return tuple((k, tuple(v.shape), v.dtype, v.device) for k, v in sorted(batch.items()) if torch.is_tensor(v))

    def __call__(self, batch):
        with torch.inference_mode():
            for k in self.keys:
                self.static[k].copy_(batch[k])
            self.graph.replay()
            out = self.out.float()
            return out.clone() if out is self.out else out  # never hand out the graph's buffer (next replay overwrites it)


def python_headers_ok():
    """Triton (torch.compile's GPU backend) builds a small C launcher and needs Python.h."""
    import os
    import sysconfig
    dirs = [sysconfig.get_paths()["include"]] + [d for d in os.environ.get("C_INCLUDE_PATH", "").split(":") if d]
    return any(os.path.isfile(os.path.join(d, "Python.h")) for d in dirs)


def install(policy, *, mode="auto", fold_bn=True, channels_last=True, amp_dtype=torch.float16,
            cudnn_benchmark=False, cuda_graph=None):
    """Speed up a loaded LeRobot ACT policy in place. Returns a dict describing what was enabled.

    mode: "compile" (torch.compile reduce-overhead + inductor freezing; fastest, slow first call),
          "graph" (manual CUDA graph; no extra requirements), "eager" (fp16 + channels_last only),
          "auto" = "compile" if Python.h is available else "graph".
    cudnn_benchmark=True buys ~3 ms in "graph" mode but autotunes with up to ~2.4 GB of transient GPU workspace
    (unified memory on a Jetson: it competes with the web app and cameras) and makes the first call take ~14 s.
    """
    if cuda_graph is not None:  # backwards-compatible flag
        mode = "graph" if cuda_graph else "eager"
    if mode == "auto":
        mode = "compile" if python_headers_ok() else "graph"
    info = {"type": policy.config.type, "mode": mode}
    if policy.config.type != "act" or not torch.cuda.is_available() or next(policy.parameters()).device.type != "cuda":
        info["enabled"] = False
        return info
    torch.backends.cudnn.benchmark = cudnn_benchmark
    backbone = getattr(policy.model, "backbone", None)
    if fold_bn and backbone is not None:
        info["bn_folded"] = fold_frozen_bn(backbone)
    if channels_last and backbone is not None:
        backbone.to(memory_format=torch.channels_last)
        info["channels_last"] = True
    info["amp"] = str(amp_dtype) if amp_dtype else None
    eager = policy.predict_action_chunk  # bound method (decorated with no_grad)

    if mode == "compile":
        import torch._inductor.config as icfg
        icfg.freezing = True  # constant-fold weights: folds BN into convs, pre-packs layouts
        patch_act_latent()  # a CPU tensor in the graph would make inductor skip CUDA graphs
        compiled = torch.compile(eager, mode="reduce-overhead")

        def fast(batch, **kw):
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
                if kw:
                    return eager(batch, **kw).float()
                b = {k: (v.contiguous(memory_format=torch.channels_last) if torch.is_tensor(v) and v.dim() == 4 else v)
                     for k, v in batch.items()}
                r = compiled(b)
                out = r.float()
                return out.clone() if out is r else out  # cudagraph-tree outputs are reused by the next replay
        policy.predict_action_chunk = fast
        info["cuda_graph"] = "inductor reduce-overhead"
        return info

    if mode == "eager":
        def fast(batch, **kw):
            with torch.inference_mode(), (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype
                                          else contextlib.nullcontext()):
                return eager(batch, **kw).float()
        policy.predict_action_chunk = fast
        info["cuda_graph"] = False
        return info

    patch_act_latent()
    state = {"g": None}

    def graphed(batch, **kw):
        if kw:  # unusual call (e.g. RTC kwargs): run eagerly
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
                return eager(batch, **kw).float()
        g = state["g"]
        if g is None or g.sig != GraphedChunk.signature(batch):
            if g is not None:
                log.warning("input shapes changed: re-capturing the CUDA graph")
            state["g"] = g = GraphedChunk(eager, batch, amp_dtype)
        return g(batch)

    policy.predict_action_chunk = graphed  # instance attribute: select_action() picks it up too
    info["cuda_graph"] = True
    return info
