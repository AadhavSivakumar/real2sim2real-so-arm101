# SO-101 web console: policy deployment — implementation spec

Code under development: `/mnt/secondary/so101_r2s/so101_web_dev/web/` (git repo; baseline commit = what runs on the
Jetson today). Read `app.py`, `recorder.py`, `index.html` first: keep their style (plain stdlib HTTP server, one control
thread that owns the serial bus, small dependency-free vanilla-JS page, dense comments only where non-obvious).

## Hard rules (safety + environment)
- The arm must never move unless the user explicitly started a non-dry-run policy episode, pressed "go to start pose",
  or used the existing teleop/manual controls. STOP (button, `Esc`, `/api/stop`) always wins and holds the pose
  (torque stays on). Every fault ends in a hold, never a limp arm or a jump.
- Testing happens ONLY on the Jetson (`ssh asnano@<jetson>`, flaky link: use `-o ConnectTimeout=30` and
  retries) in `~/so101_web_dev/` (rsync the local `web/` there) with **`SO101_MOCK=1`** (fake follower/leader/cameras,
  never opens `/dev/serial/*` or `/dev/video*`/`/dev/v4l/*`) and `SO101_DEPTH=0`, on ports **8091-8099**, with
  `--data-root ~/so101_web_dev/test_datasets` and `--policy-root ~/so101_policies`. Never touch
  `~/lerobot/so101_web/` (production; deployed later by the lead), never run the app without `SO101_MOCK=1`, never
  start/stop anything on port 8080, never `pip install`/`uv sync` into `~/lerobot/.venv` (Python: `~/lerobot/.venv/bin/python`,
  run from `~/lerobot`). Kill only processes you started (match on your port / dev path; beware `pkill -f` patterns that
  match your own ssh command line — use `pgrep -f "[a]pp.py --port 8091"` style).
- Jetson facts: Orin Nano 8 GB, 6 cores, torch 2.11 + CUDA works; `transformers` is NOT installed (SmolVLA/pi0/GR00T
  can only run on a remote GPU server). LeRobot 0.6.2 (commit b64fe1e, same as training). ACT `act_real_v0` in
  `~/so101_policies/act_real_v0` (flat pretrained_model contents: config.json, model.safetensors, train_config.json,
  policy_preprocessor*.json/.safetensors, policy_postprocessor*); load ~5-10 s (first load downloaded resnet18, now
  cached), `predict_action_chunk` = 80 ms for a (1, 50, 6) chunk (830 ms first call), CUDA mem 0.3 GB.
  A real dataset for mock camera frames: `~/so101_datasets/cube_cup_real_v1` (LeRobot v3, one mp4 per episode under
  `videos/observation.images.{wrist,scene}/chunk-000/file-00K.mp4`, parquet under `data/`).
- Units everywhere in the app: LeRobot degrees for joints 1-5, 0-100 for the gripper; joint order
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`; dataset/policy fps 30.

## Architecture
```
browser ──HTTP──> app.py (Handler) ──> PolicyRunner thread (30 Hz)  ──multiprocessing.connection──> policy_worker.py
                         │                 │  obs: ctl.snapshot() + CameraFeed.get_frame()          (subprocess on the
                         │                 │  -> undistort/map keys -> request chunks async          Jetson GPU, or a
                         │                 └─> ctl.set_policy_target(...) + heartbeat               remote GPU server)
                         └──> Controller thread (60 Hz, owns the bus): mode "policy" slews cmd toward the target
                              at max_speed; watchdogs; recorder/rollout recording read ctl.snapshot()
```

### New files
1. `policy_worker.py` — standalone policy server. `python policy_worker.py --ckpt DIR [--address unix:/path.sock |
   --address HOST:PORT] [--authkey KEY] [--device cuda]`. Loads with
   `PreTrainedConfig.from_pretrained` / `get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)` /
   `make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": dev}})`,
   warms up with a dummy observation built from `cfg.input_features`, then prints exactly one line
   `READY {json info}` on stdout and serves `multiprocessing.connection.Listener(address, authkey=...)` (one client at a
   time; loop accepting new clients; EOF = client gone). Must also run on a Brev GPU box with LeRobot (`~/lr` there),
   so no app imports. Messages (dicts):
   - `{"cmd": "info"}` -> info dict: `name, type, ckpt, device, input_features {key: shape}, output_features,
     chunk_size, n_action_steps, fps (from train_config dataset fps if available else 30), image_keys,
     deploy (contents of DIR/so101_deploy.json or None), warmup_ms`.
   - `{"cmd": "reset"}` -> `{"ok": True}` (calls policy.reset()).
   - `{"cmd": "chunk", "id": int, "task": str, "obs": {"observation.state": float32[6],
     "observation.images.<k>": ("jpeg", bytes) | uint8 HxWx3 RGB}}` -> `{"id", "actions": float32[T, 6],
     "infer_ms": float}`; errors -> `{"id", "error": str}` (never crash the server on a bad request).
     Batch construction mirrors `lerobot.common.control_utils.predict_action` (HWC uint8 -> CHW float32 /255, batch
     dim, `task`, `robot_type="so_follower"`), then `policy.predict_action_chunk(batch)` and the postprocessor on the
     (1, T, D) tensor. Only chunking policies are supported (act, smolvla, pi0, pi05, groot, ...); for others
     (diffusion, vqbet, ...) `READY` is replaced by `ERROR {json}` and exit code 2.
   - `{"cmd": "quit"}` -> exits.
2. `policy_client.py` — `PolicyClient` (connect to an address, info/reset/chunk with timeouts; JPEG-encodes images for
   non-unix addresses), `LocalWorker` (spawns policy_worker.py with a random authkey and a unix socket in
   `/tmp/so101_policy_<pid>_<n>.sock`, waits for READY/ERROR with a timeout, streams its stderr to the app log, kills it
   on unload/app exit), and `ChunkScheduler` — pure logic with an injectable clock, unit-tested:
   - `n_exec` = user override or `info.n_action_steps` (<= chunk_size).
   - episode start: one blocking chunk request with the current obs (arm holds meanwhile).
   - each 30 Hz tick: pop the next action; when `remaining <= lead` (lead = ceil(ema_infer_s * fps) + 2) and no request
     in flight, capture the obs *now* (step k) and request the next chunk asynchronously; when it arrives at step
     k + d, the new queue is `chunk[d : d + n_exec]`, cross-faded linearly with the old queue's remaining actions over
     `min(remaining, xfade_steps)` steps (default 5); if d >= len(chunk) drop it and re-request.
   - if the queue runs dry: hold the last target, count `starved` ticks; after `starve_timeout_s` (default 2.0) -> fault.
3. `policy_runner.py` — `PolicyRunner` (one per app): loads/unloads a policy (local worker or remote server listed in
   `<policy-root>/remotes.json`: `[{"name": ..., "address": "127.0.0.1:7100", "authkey": "..."}]`), runs episodes in a
   30 Hz thread, applies the per-policy deploy sidecar, handles interventions, rollout recording and outcome labels,
   keeps an eval log. Status dict (exposed in `/api/state` as `policy`, see API).
4. `mock_hw.py` — `SO101_MOCK=1` replacements used by app.py: a fake follower/leader with the attributes app.py uses
   (`bus.sync_read("Present_Position")`, `bus.calibration[j].range_min/max`, `bus.enable_torque/disable_torque`,
   `send_action`, `connect/disconnect`, `is_calibrated`), simulating first-order position tracking (~300 deg/s, small
   noise; limp = holds position); a fake camera with `frame_lock`, `latest_frame`, `latest_timestamp`, `read_latest`,
   `connect/disconnect` that loops a real episode video at 30 fps (decode with PyAV; env `SO101_MOCK_EPISODE`, default 0;
   falls back to a synthetic pattern), and the follower's initial pose = that episode's first `observation.state`.
5. `tests/` — `test_scheduler.py` (ChunkScheduler with a fake clock: prefetch timing, latency skip, crossfade, starvation,
   stale chunk), `test_worker.py` (start policy_worker on act_real_v0, info/reset/chunk shape and latency, bad request ->
   error, quit), `test_e2e.py` (start app in mock mode on a port, drive the HTTP API: load, dry run, real run on the mock
   arm, STOP mid-episode, time limit, worker killed mid-run -> hold + fault, takeover/resume, rollout recording to a test
   dataset + outcome sidecar + reload with LeRobotDataset, unload). Plain `python tests/x.py` runnable (no pytest on the
   Jetson); print a PASS/FAIL summary.

### Changes to existing files
- `app.py`: Controller gets mode `"policy"` (entered only via the runner, `/api/mode` must reject it): slews `cmd`
  toward `policy_target` at `policy_rate` (deg/s; gripper uses the same number in %/s), holds if the target is None.
  Ramp-in: until the cmd first gets within 5 deg of the target (or 2 s), cap the rate at min(rate, 45).
  Defense in depth inside the control thread: if `policy_heartbeat` (set by the runner every tick) is older than 0.5 s ->
  hold (switch to manual at the current cmd) with message "policy stalled"; tracking guard: any arm joint (not the
  gripper) with |present - cmd| > 25 deg for > 1.0 s in policy mode -> hold + fault message. `held` (recorded action)
  includes policy mode. STOP bumps `stop_count` (the runner watches it and ends the episode). Mock mode via
  `SO101_MOCK=1` (mock_hw), new args `--policy-root` (default env `SO101_POLICY_ROOT` or `~/so101_policies`).
  New endpoints (below). Shutdown: stop the runner and its worker.
- `recorder.py`: keep teleop recording unchanged in behaviour; factor out what the rollout recorder needs
  (dataset open/resume, frame add, save/discard, quality checks) instead of duplicating it. Teleop recording and policy
  episodes are mutually exclusive (clear error messages both ways).
- `index.html`: new Policy card (see UI) + joint rows show the policy target in policy mode; existing UI unchanged.
- `so101.sh`: `./so101.sh policy-server <ckpt_dir> [port]` (runs policy_worker.py on 0.0.0.0:port, authkey from
  `SO101_POLICY_KEY`, default "so101"), and a usage line.

## Deploy sidecar `<policy>/so101_deploy.json` (optional; all keys optional)
```json
{"undistort": {"wrist": {"K": [9 floats], "dist": [5 floats]}, "scene": {"K": [...], "dist": [...]}},
 "camera_map": {"observation.images.wrist": "wrist", "observation.images.scene": "scene"},
 "start_pose": [6 floats, LeRobot units],
 "task": "Put the cube in the cup", "max_speed": 120, "time_limit_s": 40, "n_action_steps": null,
 "notes": "free text shown in the UI"}
```
- undistort: `cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480), CV_16SC2)` once per camera, `cv2.remap` each
  frame before it is sent (sim-co-trained policies were trained on undistorted real frames; real-only ones on raw).
- camera_map default: each `observation.images.X` input feature maps to feed `X`; a policy whose image keys can't be
  mapped (e.g. camera1) without a camera_map and without a rename step in its preprocessor -> not runnable, with reason.
- start_pose default: `[0, -103, 95, 52, 0, 20]` (the folded rest pose the real demos start from).

## HTTP API (JSON; same token rules as the existing endpoints)
- `GET /api/policies` -> `{"root", "policies": [{"name", "path", "type", "size_mb", "steps", "dataset", "created",
  "chunk_size", "n_action_steps", "image_keys", "deploy": {...}|null, "undistort": bool, "runnable": bool,
  "why_not": str|null, "tally": {"success", "fail", "total"}}], "remotes": [{"name", "address"}]}`. A policy is a
  directory under the root containing `config.json` (also accept `<name>/pretrained_model/` and
  `<name>/checkpoints/last/pretrained_model/`). Cache by mtime; never load weights here.
- `POST /api/policy/load {"name": str}` or `{"remote": str}` -> starts loading in the background (status phase
  `loading` -> `ready` or `error`). Loading another policy unloads the current one. Refused while an episode runs.
- `POST /api/policy/unload`
- `POST /api/policy/start {"dry_run": bool, "max_speed": float, "time_limit_s": float, "n_action_steps": int|null,
  "task": str, "home_first": bool, "record": {"dataset": str}|null}` -> starts one episode. Validation: policy ready,
  no teleop recording session, torque on (mode not limp), cameras fresh, record forbidden with dry_run, dataset name
  regex as recorder. `home_first`: slow move (30 deg/s) to the start pose first (phase `homing`), then the episode.
  Dry run: the policy runs on live observations but the Controller stays in manual hold; targets only displayed.
- `POST /api/policy/stop` -> ends the episode, hold pose (the episode stays pending if recording).
- `POST /api/policy/home {"pose": [6]|null}` -> slow move to the start pose (only when no episode runs).
- `POST /api/policy/takeover` -> intervention: Controller switches to teleop (existing sync behaviour), the scheduler is
  paused, recording continues and marks intervention frames. `POST /api/policy/resume` -> back to policy mode, policy
  reset + fresh chunk from the current obs.
- `POST /api/policy/outcome {"outcome": "success"|"fail"|"none", "note": str?}` -> labels the current/last episode (ends
  it if running). If recording: pending episode saved with the label (`none` = save unlabeled); appends one line to
  `<policy>/evals.jsonl`: `{"t", "policy", "episode", "outcome", "duration_s", "interventions", "dry_run",
  "dataset", "episode_index", "settings", "note"}`; outcome labels for recorded episodes also go to
  `<dataset>/meta/so101_rollouts.jsonl` (`{"episode_index", "policy", "outcome", "intervention_frames": [[a, b], ...]}`).
- `POST /api/policy/discard` -> drop the pending (unsaved) rollout episode.
- Starting a new episode while one is pending saves it as unlabeled first.
- `GET /api/state` gains `"policy": {"phase": "none|loading|ready|homing|running|intervention|pending|error",
  "name", "backend": "local"|"remote:<addr>", "info": {...}|null, "dry_run", "episode", "elapsed_s", "time_limit_s",
  "max_speed", "infer_ms": {"last", "avg", "max"}, "chunks", "queue", "hz", "starved", "target": {joint: value}|null,
  "message", "fault": str|null, "record": {"dataset", "saved", "pending"}|null, "tally": {"success", "fail", "total"},
  "last_outcome"}` (phase `pending` = episode ended, waiting for an outcome label / discard).
- `GET /stream/<cam>.mjpg?view=policy` -> the frames as the loaded policy sees them (undistorted when its sidecar says
  so; otherwise identical to the raw stream).

## UI (index.html)
Policy card (full width, above "Record episodes"), in the existing visual language:
- Policy selector (name · type · steps · size · `undistort` badge · tally `7/10`), remote servers in an optgroup,
  Refresh, Load/Unload, status line (phase, message, backend, inference ms last/avg/max, policy Hz, queue, chunks).
- Settings: Dry run (checkbox, default ON for the first run of a policy in this page session), Max speed deg/s
  (default sidecar or 90), Time limit s (sidecar or 40), Chunk steps (blank = as trained), Task (prefilled from sidecar /
  "Put the cube in the cup"), Go to start pose first (default on), Record rollouts to dataset (optional name with the
  same datalist as the recorder).
- Big Start / Stop buttons; outcome buttons Success / Fail / Skip label / Discard (visible in `pending`, and Success/Fail
  also while running); Take over (leader) / Resume; Go to start pose. A progress bar for elapsed/time limit.
- First non-dry-run start per policy per page load asks for confirmation ("The arm will move on its own. Keep STOP /
  Esc ready.").
- Keys: `Esc` = STOP (always, global, also when an input has focus). When a policy is loaded and no teleop recording
  session is active: `Space` = stop the running episode / start the next one (start only after the first confirmed start),
  `1` = success, `2` = fail, `0` = skip label, `Backspace` = discard pending, `T` = take over, `R` = resume,
  `H` = go to start pose. The existing recording keys keep working when a teleop recording session is active; Space
  must NOT submit the teleop recording form while a policy is loaded. Show a key legend.
- Header shows a `POLICY` pill while running (`DRY RUN` while dry-running), and the mode segment highlights nothing
  (or a disabled "Policy" button) in policy mode.
- Joint rows: in policy mode show `policy <target>` next to `actual`, and slider reflects the target; in dry run the
  target is shown as a "ghost" value while the arm holds.
- Camera cards: a small toggle "policy view" switching the stream to `?view=policy` when a policy is loaded.
- Mobile width must keep working (existing media queries).
