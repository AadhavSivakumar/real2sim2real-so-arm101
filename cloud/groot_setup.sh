#!/bin/bash
# GR00T N1.7 on a fresh Brev H100 "r2s-h100b": LeRobot venv (same freeze as r2s-*) + the groot extra's deps
# (torch / lerobot / transformers pinned by the freeze), real_all_v1 aggregate (40 real demos), base model download
# and an anonymous access probe of the gated backbone repo. Log: brev/groot.log
H=${H:-r2s-h100b}; B=/mnt/secondary/so101_r2s/brev
exec >> $B/groot.log 2>&1
echo "[groot] $(date +%T) waiting for $H"
for i in $(seq 1 120); do
  timeout 20 ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no $H true 2>/dev/null && break
  [ $((i % 6)) = 0 ] && brev refresh > /dev/null 2>&1; sleep 20
done
timeout 20 ssh $H true || { echo "[groot] box not reachable"; exit 1; }
echo "[groot] $(date +%T) box up: $(ssh $H 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | grep Mem; df -h ~ | tail -1')"
grep -v "^-e " $B/lr_freeze.txt > /tmp/groot_req.txt
grep -v "^-e \|@ git+" $B/lr_freeze.txt > /tmp/groot_constraints.txt
scp -q /tmp/groot_req.txt $H:/tmp/req.txt && scp -q /tmp/groot_constraints.txt $H:/tmp/constraints.txt
ssh $H 'set -e
  c=$(readlink -f ~/.cache 2>/dev/null || true); [ -n "$c" ] && [ -d "$c" ] && [ ! -w "$c" ] && sudo chown -R $USER "$c"
  for t in $(seq 1 40); do command -v ffmpeg > /dev/null && break; sudo apt-get -o DPkg::Lock::Timeout=600 update -qq && sudo apt-get -o DPkg::Lock::Timeout=600 install -y -qq ffmpeg > /dev/null && break; sleep 15; done; command -v ffmpeg > /dev/null
  export PATH=$HOME/.local/bin:$PATH; command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
  [ -x ~/lr/bin/python ] || uv venv --python 3.12 --seed ~/lr -q
  uv pip install --python ~/lr/bin/python -q -r /tmp/req.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  uv pip install --python ~/lr/bin/python -q -c /tmp/constraints.txt "peft>=0.18.0,<1.0.0" "diffusers>=0.38.0,<0.40.0" "dm-tree>=0.1.8,<1.0.0" "timm>=1.0.0,<1.1.0" "decord>=0.6.0,<1.0.0" \
     --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  ~/lr/bin/python -c "import lerobot, torch, torchcodec, transformers, peft, diffusers, tree, timm, decord; import lerobot.policies.groot; print(\"env ok\", lerobot.__version__, torch.__version__, transformers.__version__, peft.__version__, diffusers.__version__, timm.__version__, torch.cuda.is_available())"
  ~/lr/bin/pip freeze 2>/dev/null | grep -iE "^(torch|torchvision|torchcodec|transformers|lerobot|peft|diffusers|timm|dm-tree|decord)[ =@]"
  mkdir -p ~/r2s/lerobot_datasets ~/r2s/train_runs ~/r2s/logs' || { echo "[groot] setup failed"; exit 1; }
echo "[groot] $(date +%T) env ok; uploading real data"
rsync -a --delete /mnt/secondary/so101_r2s/real_datasets/cube_cup_real_v2_merged/ $H:r2s/lerobot_datasets/cube_cup_real_v2/ && \
rsync -a /mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1/ $H:r2s/lerobot_datasets/real_cube_cup_v1/ || { echo "[groot] upload failed"; exit 1; }
ssh $H 'cd ~/r2s && rm -rf lerobot_datasets/real_all_v1 && ~/lr/bin/python - <<PY
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
R = Path.home() / "r2s/lerobot_datasets"
aggregate_datasets(["local/real_cube_cup_v1", "local/cube_cup_real_v2"], "local/real_all_v1",
                   roots=[R / "real_cube_cup_v1", R / "cube_cup_real_v2"], aggr_root=R / "real_all_v1")
ds = LeRobotDataset("local/real_all_v1", root=R / "real_all_v1")
print("real_all_v1:", ds.meta.total_episodes, "episodes", ds.meta.total_frames, "frames", list(ds.meta.tasks.index))
PY' > /tmp/groot_aggr.log 2>&1; grep -E "real_all_v1|Error|Traceback" /tmp/groot_aggr.log
grep -q "^real_all_v1: 40 episodes" /tmp/groot_aggr.log || { echo "[groot] aggregate failed"; exit 1; }
echo "[groot] $(date +%T) downloading nvidia/GR00T-N1.7-3B and probing nvidia/Cosmos-Reason2-2B (anonymous)"
ssh $H 'cd ~/r2s && ~/lr/bin/python - <<PY
from huggingface_hub import snapshot_download, hf_hub_download, get_token
print("hf token present on box:", get_token() is not None)
p = snapshot_download("nvidia/GR00T-N1.7-3B"); print("GR00T-N1.7-3B ->", p)
try:
    hf_hub_download("nvidia/Cosmos-Reason2-2B", "config.json"); print("COSMOS_ACCESS ok")
except Exception as e:
    print("COSMOS_ACCESS fail", type(e).__name__, str(e).splitlines()[0][:200])
PY' 2>&1 | grep -v "it/s\]" | tail -6
echo "[groot] $(date +%T) setup done"
