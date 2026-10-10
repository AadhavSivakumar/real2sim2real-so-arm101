#!/bin/bash
# New Brev sim box (Ubuntu 22.04, driver >= 570): Isaac Sim 6.1 pip wheels + IsaacLab 3.0 (a8b4da3) in the uv venv ~/isl,
# same layout as r2s-sim. Then copy the repo (with the remote lab.sh) and /mnt/secondary/so101_r2s/assets over.
set -ex
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
for i in $(seq 1 40); do sudo apt-get -o DPkg::Lock::Timeout=60 update -qq && break; sleep 15; done
sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y -qq libvulkan1 ffmpeg cmake git libglu1-mesa libxt6 libxrender1 \
  libxi6 libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 libegl1 libgl1 > /dev/null
sudo mkdir -p /mnt/secondary/so101_r2s && sudo chown -R "$USER" /mnt/secondary
uv venv --python 3.12 --seed ~/isl
source ~/isl/bin/activate
uv pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
[ -d ~/IsaacLab ] || git clone -q https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && git checkout -q a8b4da3c29ae528b39d4b3c9444d782ce58d886d
OMNI_KIT_ACCEPT_EULA=yes ./isaaclab.sh -i
uv pip install "moviepy>=1.0.3,<2.0.0.dev0"
echo SETUP_SIM_DONE
