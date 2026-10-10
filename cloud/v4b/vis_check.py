"""Overlay the FK-estimated cup / cube positions on overhead (scene.mp4) frames of raw v3 episodes."""
import json, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, '/mnt/secondary/so101_r2s/brev')
CAL = json.load(open('/home/asubuntudesktop/TestingGrounds/so101_r2s2r/calib/data/calib.json'))
K = np.array(CAL['SCENE_K']).reshape(3, 3); C = np.array(CAL['SCENE_CAM_POS']); x, y, z, w = CAL['SCENE_CAM_QUAT']
R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
DESK = np.array([(-0.12, 0.60), (-0.12, -0.226), (0.386, -0.164), (0.339, 0.60)])
def proj(p):
    pc = R.T @ (np.asarray(p, float) - C); return int(K[0,0]*pc[0]/pc[2]+K[0,2]), int(K[1,1]*pc[1]/pc[2]+K[1,2])
def frame(path, t):
    raw = subprocess.run(['ffmpeg','-v','error','-i',path,'-vf',f'select=eq(n\\,{t})','-vframes','1','-f','rawvideo','-pix_fmt','bgr24','-'],capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(480, 640, 3).copy()
pick = json.load(open(sys.argv[1])); tiles = []
for key, rs in pick.items():
    for r in rs:
        ep = f"/mnt/secondary/so101_r2s/sim_demos/{r['source']}/{r['ep']}"
        im = frame(ep + '/scene.mp4', 0)
        for i in range(4):
            a, b = DESK[i], DESK[(i+1)%4]; cv2.line(im, proj([*a, 0]), proj([*b, 0]), (0,255,255), 1)
        cx, cy = r['cup_xy']
        for zz, col in ((0.0, (255,0,255)), (0.095, (255,0,0))):
            pts = [proj([cx+0.04*np.cos(t), cy+0.04*np.sin(t), zz]) for t in np.linspace(0, 2*np.pi, 25)]
            cv2.polylines(im, [np.array(pts)], True, col, 1)
        if r.get('cube_xy'):
            cv2.drawMarker(im, proj([*r['cube_xy'], 0.015]), (0,0,255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(im, f"{key} {r['source'][10:]}/{r['ep'][-3:]} d={r['cup_d']:+.3f}", (5, 20), 0, 0.55, (255,255,255), 2)
        tiles.append(im)
rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(sys.argv[2], cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6))
