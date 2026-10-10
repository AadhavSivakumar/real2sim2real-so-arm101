"""Top-down cube-position map (robot frame, overhead-camera orientation) as an inline SVG using the page's CSS tokens."""
import json
import numpy as np
a = np.load("train_positions.npy"); sim = a[a[:, 0] < 4013][:, 1:3]
real = np.load("/mnt/secondary/so101_r2s/videos_v4/desk_check/real_positions.npz")["cube"][:, :2]
runs = json.load(open("rollout_positions.json"))
FAIL = (0.104, -0.145)
YMAX, XMAX = 0.42, 0.40          # view: robot y in [-0.26, 0.42] (left edge of the image = robot's left), x in [-0.13, 0.40]
W, H = 680, 530
P = lambda x, y: (round((YMAX - y) * 1000, 1), round((XMAX - x) * 1000, 1))
desk = ((-0.12, 0.60), (-0.12, -0.226), (0.386, -0.164), (0.339, 0.60))
o = [f'<svg class="cov" viewBox="0 0 {W} {H}" role="img" aria-label="Top-down map of cube positions: sim demos cover the whole desk, real demos lean left, and the real-only failure sits where real demos are sparse">',
     f'<clipPath id="covclip"><rect width="{W}" height="{H}"/></clipPath><g clip-path="url(#covclip)">',
     '<polygon points="' + " ".join("%s,%s" % P(*p) for p in desk) + '" fill="var(--surface)" stroke="var(--line)" stroke-width="2"/>']
x0, y0 = P(0, 0)
for deg in (-60, -30, 0, 30, 60):   # angle guides from the base
    t = np.radians(deg); x1, y1 = P(0.36 * np.cos(t), 0.36 * np.sin(t))
    o.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="var(--line)" stroke-width="1.5" stroke-dasharray="4 5"/>')
rng = np.random.default_rng(0); keep = rng.permutation(len(sim))[:2400]
o.append('<g fill="var(--sim)" fill-opacity="0.28">' + "".join('<circle cx="%s" cy="%s" r="2.6"/>' % P(*p) for p in sim[keep]) + '</g>')
o.append('<g fill="var(--real)" stroke="var(--surface)" stroke-width="1.5">' + "".join('<circle cx="%s" cy="%s" r="6"/>' % P(*p) for p in real) + '</g>')
fx, fy = P(*FAIL)
o.append(f'<circle cx="{fx}" cy="{fy}" r="50" fill="none" stroke="var(--bad)" stroke-width="2" stroke-dasharray="6 5"/>')
for r in runs:
    x, y = P(r["x"], r["y"]); col = "var(--good)" if r["outcome"] == "success" else "var(--bad)"
    letter = "R" if "real40" in r["ds"] else "C"
    o.append(f'<circle cx="{x}" cy="{y}" r="11" fill="var(--surface)" stroke="{col}" stroke-width="3"/>'
             f'<text x="{x}" y="{y + 4.5}" text-anchor="middle" font-size="13" font-weight="700" fill="{col}" font-family="var(--f-mono)">{letter}</text>')
o.append(f'<rect x="{x0 - 22}" y="{y0 - 4}" width="44" height="40" rx="5" fill="var(--comp)"/>'
         f'<text x="{x0}" y="{y0 + 21}" text-anchor="middle" font-size="12" fill="var(--surface)" font-family="var(--f-mono)">base</text>')
o.append(f'<text x="14" y="26" font-size="14" fill="var(--muted)" font-family="var(--f-mono)">robot\'s left</text>'
         f'<text x="{W - 14}" y="26" text-anchor="end" font-size="14" fill="var(--muted)" font-family="var(--f-mono)">robot\'s right</text>')
o.append('</g></svg>')
svg = "".join(o)
open("coverage.svg.html", "w").write(svg)
print(len(svg), "bytes;", len(keep), "of", len(sim), "sim points drawn")
