"""ChArUco boards used for camera intrinsics. Lens calibration does not depend on the board's absolute
size, so a board shown on a phone/tablet/laptop screen works as well as paper (no ruler needed).

  paper: page 2 of so101_calibration_targets.pdf (5x7 squares, DICT_5X5_100)
  phone: so101_phone_board.png shown full-screen (4x7 big squares, DICT_4X4_50 ids 10-23 so it never
         collides with the desk anchor, which is DICT_4X4_50 id 0)
"""
import cv2
import numpy as np

BOARDS = {
    "paper": cv2.aruco.CharucoBoard((5, 7), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)),
    "phone": cv2.aruco.CharucoBoard((4, 7), 0.016, 0.012, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                    np.arange(10, 24)),
}
MIN_CORNERS = {"paper": 12, "phone": 8}  # of 24 / 18 inner corners


def _detector(board):
    """ChArUco detector tuned for small / slightly blurred boards (a phone seen by the overhead camera)."""
    dp = cv2.aruco.DetectorParameters()
    dp.adaptiveThreshWinSizeMin, dp.adaptiveThreshWinSizeMax, dp.adaptiveThreshWinSizeStep = 3, 63, 6
    dp.minMarkerPerimeterRate, dp.polygonalApproxAccuracyRate, dp.errorCorrectionRate = 0.01, 0.05, 0.8
    cp = cv2.aruco.CharucoParameters()
    cp.minMarkers, cp.tryRefineMarkers = 1, True
    return cv2.aruco.CharucoDetector(board, cp, dp)


def detect(gray, board) -> tuple:
    """(corners Nx1x2 float32 in original pixels, ids Nx1) or (None, None): the better of OpenCV's default
    detector and a tuned one on a 2x upscale (which roughly doubles the usable frames when the markers are
    only ~15 px), falling back to a plain chessboard search when neither finds every corner."""
    cc0, ci0, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cc1, ci1, _, _ = _detector(board).detectBoard(up)
    n0, n1 = (0 if ci0 is None else len(ci0)), (0 if ci1 is None else len(ci1))
    nx, ny = board.getChessboardSize()
    if max(n0, n1) < (nx - 1) * (ny - 1):
        cc2 = _chessboard(gray, (nx - 1, ny - 1))
        if cc2 is not None:
            return cc2, np.arange(len(cc2), dtype=np.int32).reshape(-1, 1)
    if max(n0, n1) == 0:
        return None, None
    if n0 >= n1:
        return cc0, ci0
    return ((cc1 + 0.5) / 2 - 0.5).astype(np.float32), ci1


def _chessboard(gray, pattern):
    """All inner corners from the board's black squares alone, or None. Rescues frames where a bright phone
    screen washes out the thin marker borders (the squares survive). Row-major like ChArUco corner ids; the
    start corner is ambiguous, but any grid symmetry is just another valid board pose for lens calibration.
    Uses the classic detector: OpenCV 5.0's findChessboardCornersSB returns scrambled 'grids' on this board.
    A homography fit rejects anything that is not a clean planar grid."""
    grid = np.array([[x, y] for y in range(pattern[1]) for x in range(pattern[0])], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    for s in (1, 2):
        g = gray if s == 1 else cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        ok, c = cv2.findChessboardCorners(g, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit)
        c = ((c + 0.5) / s - 0.5).astype(np.float32)
        H, _ = cv2.findHomography(grid, c.reshape(-1, 2))
        if H is not None and np.abs(cv2.perspectiveTransform(grid[None], H)[0] - c.reshape(-1, 2)).max() < 1.0:
            return c
    return None
