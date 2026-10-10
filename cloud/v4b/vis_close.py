"""Overlay projected TCP / fingertip at the first CLOSE and last LOWER-RELEASE frames of raw episodes."""
import json, subprocess, sys, importlib.util
import numpy as np, cv2
exec(open('/mnt/secondary/so101_r2s/brev/v4b/vis_check.py').read().split('pick = json.load')[0])
s = importlib.util.spec_from_file_location('flag', '/mnt/secondary/so101_r2s/brev/v4b_flag.py'); F = importlib.util.module_from_spec(s); s.loader.exec_module(F)
tiles = []
for spec in sys.argv[2:]:
    src, ep = spec.split('/')
    d = np.load(f'/mnt/secondary/so101_r2s/sim_demos/{src}/{ep}/data.npz'); q, ph = d['state'], d['phase']
    cl = F.segments(ph, 2); rel = [(a, b) for a, b in F.segments(ph, 6) if ph[a-1] == 5]
    for t, lab in ((0, 'f0'), (cl[0][0], 'close0'), (rel[-1][1]-1, 'rel_end')):
        im = frame(f'/mnt/secondary/so101_r2s/sim_demos/{src}/{ep}/scene.mp4', t)
        for i in range(4):
            a, b = DESK[i], DESK[(i+1)%4]; cv2.line(im, proj([*a, 0]), proj([*b, 0]), (0,255,255), 1)
        tcp, cube, tip = F.hand(q[t].astype(float), float(d['cube_size']))
        cv2.drawMarker(im, proj(tcp), (0,0,255), cv2.MARKER_CROSS, 12, 2)
        cv2.drawMarker(im, proj(tip), (255,0,0), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.drawMarker(im, proj([tcp[0], tcp[1], 0]), (0,255,0), cv2.MARKER_SQUARE, 10, 1)
        cv2.putText(im, f"{src[10:]}/{ep[-3:]} {lab} t={t}", (5, 20), 0, 0.55, (255,255,255), 2)
        tiles.append(im)
rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(sys.argv[1], cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6))
