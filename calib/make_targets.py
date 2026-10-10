"""Printable calibration targets (PDF, 2 pages, fits US Letter and A4 at 100% scale).

Page 1: ArUco DICT_4X4_50 id 0, 100 mm black square -> desk anchor (metric scale + sim frame).
Page 2: ChArUco board, DICT_5X5_100, 5x7 squares of 30 mm, 22 mm markers -> camera intrinsics.
Also writes so101_phone_board.png next to the PDF: the "phone" board (boards.py) at phone resolution, to show
full-screen instead of printing page 2.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MM = 1 / 25.4  # inches per mm
PAGE_W, PAGE_H = 8.27, 11.0  # inches: min(A4, Letter) per side, so it prints on both


def page(pdf, img, size_mm, title, notes):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    w, h = size_mm
    ax = fig.add_axes([(PAGE_W - w * MM) / 2 / PAGE_W, (PAGE_H - h * MM) / 2 / PAGE_H, w * MM / PAGE_W, h * MM / PAGE_H])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    fig.text(0.5, 0.95, title, ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.04, notes, ha="center", fontsize=8.5, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


d4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(d4, 0, 600, borderBits=1)
marker = cv2.copyMakeBorder(marker, 75, 75, 75, 75, cv2.BORDER_CONSTANT, value=255)  # white quiet zone

d5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((5, 7), 0.030, 0.022, d5)
board_img = board.generateImage((1500, 2100), marginSize=0, borderBits=1)

with PdfPages(sys.argv[1]) as pdf:
    page(pdf, marker, (125, 125), "SO-101 desk anchor - ArUco 4x4 id 0",
         "Print at 100% / 'Actual size' (no fit-to-page). The BLACK square must measure exactly 100 mm.\n"
         "Tape it flat on the desk, fully visible to the overhead Astra, outside the arm's working area. Never move it afterwards.")
    page(pdf, board_img, (150, 210), "Camera calibration board - ChArUco 5x7, 30 mm squares",
         "Print at 100%. Each square must measure exactly 30 mm. Tape onto stiff cardboard so it stays perfectly flat.")
print("wrote", sys.argv[1])

from boards import BOARDS  # noqa: E402

W, H = 1170, 2532  # common phone portrait resolution; any screen works (the board scales with it)
phone = BOARDS["phone"].generateImage((1000, 1750), marginSize=0, borderBits=1)
phone = cv2.copyMakeBorder(phone, (H - 1750) // 2, (H - 1750) // 2, (W - 1000) // 2, (W - 1000) // 2,
                           cv2.BORDER_CONSTANT, value=255)
out = Path(sys.argv[1]).with_name("so101_phone_board.png")
cv2.imwrite(str(out), phone)
print("wrote", out)
