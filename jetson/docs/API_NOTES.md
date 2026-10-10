# API notes (backend) — clarifications and additions to SPEC.md

Nothing in SPEC.md's API was renamed or removed. These are the choices the spec left open, plus extra fields.

## Behaviour choices
- **Every episode ends in phase `pending`**, dry runs and unrecorded episodes included. A label (`outcome`) appends
  one line to `evals.jsonl`; dry runs are logged with `"dry_run": true` and never count in the tally. Starting a new
  episode labels the pending one `none` first. `discard` drops the pending rollout and writes **no** eval line.
- `outcome` / `discard` while an episode runs: the episode ends first (arm holds), then the label is applied.
- **`home_first` is ignored for dry runs**: a dry run never moves the arm. Use `/api/policy/home` before it if needed.
- **`unload` is refused while an episode or homing runs**, like `load`. A pending episode survives load/unload;
  labeling it still writes to the policy it ran with.
- `/api/policy/home` works with no policy loaded (default pose `[0,-103,95,52,0,20]`, else the sidecar's
  `start_pose`). It is refused while an episode runs, while a teleop recording session is active, or when the arm is
  limp. Phase is `homing` meanwhile; STOP / `/api/policy/stop` interrupt it.
- `/api/policy/stop` when nothing runs: 200 with `"nothing is running"`. It is idempotent.
- Changing the mode with `/api/mode` during a non-dry episode ends it: manual/teleop → the arm holds or follows that
  mode, limp → limp. `/api/mode {"mode":"policy"}` → 400.
- Takeover is refused in a dry run (the policy isn't moving the arm) and when the leader is missing.
- Phase `error` = load failed or the worker died. `message` says why; start is refused until the policy is reloaded.
- Start is refused (400) while a labeled rollout is still being saved (`saving: true`), usually under a second or two.
- Teleop recording `/api/record/start` is refused while an episode runs, while homing, while saving, or while a
  **recorded** rollout awaits its label. A pending unrecorded episode does not block it.

## Field formats
- `last_outcome`: string `"success" | "fail" | "none" | "discarded"` or `null`.
- `tally`: `{"success", "fail", "total"}` with `total = success + fail` (non-dry-run). Episodes run in mock mode carry
  `"mock": true` in evals.jsonl and only count when the app itself runs in mock mode.
- `intervention_frames`: half-open `[first, end)` frame-index ranges of the recorded episode.
- `record` (status): `{"dataset", "saved" (episodes saved this app session), "pending" (frames recorded and not saved
  yet), "frames"}`.
- `target`: the raw policy action for this tick (unclamped). During homing it is the start pose; in a dry run it is
  the "ghost". It is `null` while waiting for the first chunk and during a takeover.

## Extra fields
- `/api/state` → `policy` also has: `rtt_ms` (request→reply EMA incl. transport), `late` (loop ticks late), `dropped`
  (chunks that arrived too late), `errors`, `interventions`, `start_pose` [6], `n_action_steps`, `task`, `saving`,
  `backend_ok`, `backend_error`, `deploy` (sidecar of the loaded policy), `undistort` (list of undistorted cameras).
- `/api/state` top level also has `stop_count` and `mock` (true when running with SO101_MOCK=1).
- `/api/policies` → each policy also has `ckpt` (resolved pretrained_model dir) and `camera_map`.
- `policy.info` (worker info) also has `host`, `image_sizes`, `state_dim`, `action_dim`, `other_inputs`,
  `rename_map`, `load_s`, `lerobot_version`.
- Worker protocol: `{"cmd": "ping"}` → `{"ok": true, "t": ...}` (used for idle health checks).

## Files
- Remote policies log evals to `<policy-root>/_remote/<remote name>/evals.jsonl`. `_`-prefixed directories are not
  listed as policies.
- `<dataset>/meta/so101_rollouts.jsonl` gets a line for every saved rollout, including `outcome: "none"`, with
  `episode_index, policy, outcome, intervention_frames, t, episode, frames, duration_s, end_reason, fault, note,
  settings, quality`.
- Tests use `--policy-root ~/so101_web_dev/test_datasets/_e2e_policies` (symlinks to act_real_v0 plus fixtures), so
  test runs never write into `~/so101_policies`.

## Precision (added at integration, 2026-10-09)
- `policy_worker.py` runs the model call (`predict_action_chunk`) under `torch.autocast("cuda", dtype=float16)` by
  default; weights and the pre/post-processors stay fp32, and the actions are cast back to fp32. `--fp32` (or env
  `SO101_POLICY_FP32=1`, which the app's LocalWorker inherits) turns it off. `info.precision` is `"fp16-autocast"` or
  `"fp32"` and shows in the Policy card and the Ready message.
- act_real_v0 on the Orin Nano (worker alone, random 640x480 frames, N=60): fp32 p50 71.8 / p95 73.9 ms, warm-up
  672 ms; fp16 p50 56.9 / p95 57.3 ms, warm-up 329 ms. The same observation gives actions within 2.0 (max) /
  0.43 (mean) LeRobot units of the fp32 ones.

## Safety-review changes (2026-10-09)
- **`/api/policy/start` defaults to a dry run** when `dry_run` is missing (the page always sends it).
- **Takeover** is refused (400) unless the leader is within 12 deg of the follower on every arm joint:
  "bring the leader to the follower's pose first: <joint> is N deg away (max 12)". The runner re-checks it, plus STOP
  and pending mode changes, atomically in `Controller.request_takeover(stop_count)`.
- STOP always wins: the runner checks `stop_count` first every tick; `stop()` always requests manual (overriding a
  pending switch into policy/teleop); the control thread re-checks `stop_count` right before entering policy mode;
  `enter_policy` refuses while another mode change is pending; ending an episode cancels a queued takeover; if the
  controller hasn't left policy mode 0.25 s after an episode ends, the runner presses STOP.
- Follower read failures: 6 in a row in policy mode -> hold + fault "Follower reads failing"; the runner faults
  "control loop stalled" if the control tick doesn't advance for 0.25 s.
- After the policy starved, the next chunk is cross-faded from the held action and the controller ramps in again.
- Chunk request ids are unique for the process; the client's generation check + result delivery share a lock with
  `cancel()`.
- `/api/target` returns 400 for non-finite or non-numeric values.
- Page: no `alert()` left (errors go to the Policy card's flash line); the record form and "Release torque" are
  disabled during homing/running/takeover; Space starts an episode only after the phase has been ready/pending for 1 s.
- `SO101_MOCK` accepts 1/true/yes/on. Without mock mode, `app.py` refuses ports other than 8080 unless `--real`, and
  holds an exclusive flock on `/tmp/so101_follower.lock` (a second console exits with a clear message).
- Mock: `SO101_MOCK_LEADER=follow` makes the mock leader report the follower's position (a person matching the pose),
  so takeovers pass the gap check; test_e2e's app uses it. The default mock leader still rests at the start pose.

## ACT acceleration in the worker (2026-10-09)
- `web/jetson_accel.py` (copy of `accel/jetson_accel.py`) is applied by `policy_worker.py` to ACT on CUDA:
  folded BatchNorm, channels_last backbone, fp16 autocast, whole `predict_action_chunk` in one CUDA graph (captured by
  the load-time warm-up). `SO101_ACCEL=graph` (default) | `auto` (= graph here: no Python.h) | `compile` (falls back
  to graph without Python.h; a real compile takes ~250 s cold, over `LOAD_TIMEOUT_S` = 300 s margin) | `eager` | `off`.
  Any failure in install() or in the warm-up capture falls back to plain fp16 autocast (`info.accel.error` says why).
  `--fp32` = no acceleration, fp32. `info.precision`: `fp16+cudagraph` | `fp16-autocast` | `fp32`; `info.accel` has
  the details.
- act_real_v0 on the Jetson, `Policy.predict` (prep + model + post + .cpu()), 100 calls over 12 real frames:
  fp32 p50 72.2 / p95 74.2 ms; fp16 autocast 62.5 / 68.5 ms; fp16+cudagraph 40.2 / 40.6 ms. Max |action - fp32|
  0.052 (graph), 0.054 (fp16 autocast). Graph adds ~26 MB allocated GPU memory; capture adds ~0.1 s to warm-up.
