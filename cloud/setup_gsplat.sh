#!/bin/bash
# r2s-real: gsplat 1.6 (editable from ~/splat/src/gsplat, built for sm_86 with CUDA 12.8) + pycolmap in the uv venv ~/gs.
set -ex
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=24
uv venv --python 3.12 --seed ~/gs
source ~/gs/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install ninja numpy scipy scikit-learn tqdm "torchmetrics==1.8.2" opencv-python-headless "tyro>=0.8.8,!=1.0.9,!=1.0.10" \
  Pillow piexif tensorboard tensorly pyyaml matplotlib splines viser "imageio[ffmpeg]" plyfile "pycolmap>=3.10.0" \
  "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
uv pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
cd ~/splat/src/gsplat && rm -rf build && uv pip install --no-build-isolation -e .
python -c "import gsplat, torch; from gsplat.cuda._backend import _C; print('GSPLAT_OK', gsplat.__version__, torch.cuda.get_device_name())"
