# SO-101 cube-into-cup: real2sim2real

Code: `~/TestingGrounds/so101_r2s2r` · data/assets/runs: `/mnt/secondary/so101_r2s`

| Piece | Where |
|---|---|
| Isaac Lab tasks `SO101-CubeInCup-v0` (state), `-Cam-v0` (wrist + overhead 640x480, splat background), `-RL-v0` (PPO) | `so101_r2s2r/` |
| Scripted expert (94% success in sim) | `so101_r2s2r/expert.py`, eval: `scripts/run_expert.py` |
| Demo recorder → LeRobot dataset (same schema as the Jetson web-console recorder) | `scripts/record_demos.py`, `scripts/convert_to_lerobot.py` |
| Calibration (camera intrinsics, overhead camera pose, splat alignment) | `calib/` → `calib/data/calib.json` (read by `so101_r2s2r/calib.py`) |
| Sim ↔ LeRobot units | `so101_r2s2r/units.py` |

Always launch Isaac Lab through `./lab.sh` (it clears conda, which `isaaclab.sh` refuses to run inside).
`PY=~/TestingGrounds/robot_splat/.venv/bin/python` (numpy/opencv/gsplat) · `LR=/mnt/secondary/so101_r2s/.venv-lerobot/bin/python` (LeRobot @ the Jetson's commit b64fe1e)

## 1. Calibrate (needs the real rig; never moves the arm by itself)

Start the web console on the Jetson (`cd ~/lerobot && ./so101.sh web`), then on the desktop, from this folder:

1. **Lens intrinsics**, page 2 board on cardboard, wave it in front of each camera:
   `$PY calib/capture.py intrinsics --cam scene` → `$PY calib/solve.py intrinsics --cam scene` (repeat with `--cam wrist`).
   Aim for RMS < 0.5 px.
2. **Anchor**: page 1 marker taped flat on the desk, visible to the overhead camera.
   `$PY calib/capture.py snapshot --name anchor`
   `$PY calib/capture.py touch` — with the leader arm, touch the closed jaw tips to the black square's corners in the order
   **top-left, top-right, bottom-right, bottom-left as printed** (title text = top), Enter after each.
   `$PY calib/solve.py scene --snap anchor` → overhead camera pose. Touched heights should read ~0 mm and the square ~100 mm;
   if not, the sim↔real joint mapping needs attention.
3. **Background splat**: phone video of the cleared desk (marker in place, arm at rest), then the
   `~/TestingGrounds/robot_splat` pipeline (frames → `scripts/run_sfm.py` → gsplat `simple_trainer.py`), then
   `$PY calib/solve.py splat --scene <scene dir> --ply <point_cloud_29999.ply>` → aligned, robot-cropped `background.usdc`.

## 2. Simulate and generate data

```bash
./lab.sh -p scripts/run_expert.py --num_envs 64 --episodes 256 --viz none          # expert success rate
./lab.sh -p scripts/record_demos.py --episodes 500 --preview 4 --out /mnt/secondary/so101_r2s/sim_demos/run1 --viz none
$LR scripts/convert_to_lerobot.py --src /mnt/secondary/so101_r2s/sim_demos/run1 --name sim_cube_in_cup_v1
```

## 3. Train

- Imitation (ACT): `/mnt/secondary/so101_r2s/.venv-lerobot/bin/lerobot-train --dataset.repo_id=local/<name> --dataset.root=/mnt/secondary/so101_r2s/lerobot_datasets/<name> --policy.type=act --dataset.video_backend=pyav --policy.push_to_hub=false --wandb.enable=false`
- Co-training: record real demos with the web console (task text exactly **"Put the cube in the cup"**), copy the dataset folder here, and train on sim + real together.
- RL (privileged state, PPO): from `/mnt/secondary/so101_r2s/rl`:
  `~/TestingGrounds/so101_r2s2r/lab.sh train --rl_library rsl_rl --external_callback so101_r2s2r.register_tasks --task SO101-CubeInCup-RL-v0 --num_envs 512 --viz none`

## Known gaps

- Gripper 0-100 ↔ radians mapping is linear between the sim jaw limits; verify against the real jaw.
- Overhead camera is a pinhole in sim; if the Astra's distortion is large, undistort real frames or add lens distortion in sim.
- Cup is a placeholder (80/60 mm diameter, 95 mm tall) until measured — `so101_r2s2r/cup.py`.
