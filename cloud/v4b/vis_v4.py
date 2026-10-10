"""Contact sheet: cotrain_v4 sim episodes (overhead camera, first frame) with the FK-estimated cup (circle) / cube (cross)
and the desk polygon (yellow), for dropped and kept episodes. Uses the local copy of cotrain_v4 + v4b_flag.py's table."""
import json, subprocess, sys
import numpy as np, pandas as pd, cv2
exec(open('/mnt/secondary/so101_r2s/brev/v4b/vis_check.py').read().split('def frame')[0])
L = '/mnt/secondary/so101_r2s/lerobot_datasets/cotrain_v4'
src = json.load(open(f'{L}/meta/v4_sources.json'))
flags = [json.loads(l) for l in open('/mnt/secondary/so101_r2s/brev/v4b/flags.jsonl')]
by = {}
for r in flags: by.setdefault(r['source'], []).append(r)
for v in by.values(): v.sort(key=lambda r: r['ep'])
order = []
for part in src['sims']:
    s, k = part[4:].rsplit('_', 1); P = 40 if s.startswith('sim_v4_') else (84 if int(s.rsplit('_s', 1)[1]) <= 4 else 63)
    order += by[s][int(k) * P:(int(k) + 1) * P]
eps = pd.read_parquet(f'{L}/meta/episodes/chunk-000/file-000.parquet').sort_values('episode_index').reset_index(drop=True)
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
groups = [('DROP cup off', lambda r: 'cup_off_desk' in r['reasons']), ('DROP cup edge', lambda r: 'cup_at_edge' in r['reasons']),
          ('DROP cube off', lambda r: r['reasons'] == ['cube_off_desk']), ('KEEP', lambda r: not r['invalid'])]
tiles = []
for name, cond in groups:
    idx = [i for i, r in enumerate(order) if cond(r) and r['source'].startswith('sim_v4')]
    for i in rng.choice(idx, 3, replace=False):
        r, e = order[i], eps.loc[i]
        f = f"{L}/videos/observation.images.scene/chunk-{int(e['videos/observation.images.scene/chunk_index']):03d}/file-{int(e['videos/observation.images.scene/file_index']):03d}.mp4"
        t = float(e['videos/observation.images.scene/from_timestamp']) + 0.5 / 30
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.4f}', '-i', f, '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
        im = np.frombuffer(raw[:480 * 640 * 3], np.uint8).reshape(480, 640, 3).copy()
        for j in range(4):
            a, b = DESK[j], DESK[(j + 1) % 4]; cv2.line(im, proj([*a, 0]), proj([*b, 0]), (0, 255, 255), 2)
        cx, cy = r['cup_xy']
        for zz, col in ((0.0, (255, 0, 255)), (0.095, (255, 0, 0))):
            pts = [proj([cx + 0.04 * np.cos(a), cy + 0.04 * np.sin(a), zz]) for a in np.linspace(0, 2 * np.pi, 25)]
            cv2.polylines(im, [np.array(pts)], True, col, 2)
        if r.get('cube_xy'):
            cv2.drawMarker(im, proj([*r['cube_xy'], 0.015]), (0, 0, 255), cv2.MARKER_CROSS, 18, 3)
        cv2.rectangle(im, (0, 0), (640, 30), (0, 0, 0), -1)
        cv2.putText(im, f"{name}  v4 ep {i}  cup edge d={r['cup_d']:+.3f} m", (6, 22), 0, 0.7, (255, 255, 255), 2)
        tiles.append(im)
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(sys.argv[1], cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
