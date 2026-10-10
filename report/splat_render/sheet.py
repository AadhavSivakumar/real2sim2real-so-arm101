import cv2, numpy as np, glob, sys
fs = sorted(glob.glob(sys.argv[1] + '/f_*.png'))
if len(fs) > 8:
    fs = [fs[int(round(k))] for k in np.linspace(0, len(fs) - 1, 8)]
ims = [cv2.imread(f) for f in fs]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 480
ims = [cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA) for im in ims]
lab = sys.argv[4] if len(sys.argv) > 4 else 'frame'
for f, im in zip(fs, ims):
    i = int(f.split('_')[-1][:4])
    cv2.putText(im, f"{lab} {i} ({i / 30:.1f}s)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, f"{lab} {i} ({i / 30:.1f}s)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
g = np.concatenate([np.concatenate(ims[:4], 1), np.concatenate(ims[4:8], 1)], 0)
cv2.imwrite(sys.argv[2], g, [cv2.IMWRITE_JPEG_QUALITY, 90])
