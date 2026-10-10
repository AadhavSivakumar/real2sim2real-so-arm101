#!/usr/bin/env python
"""Orbbec Astra depth capture, run as a subprocess of the web app.

Runs out-of-process because the Orbbec OpenNI2 driver can segfault when the device misbehaves.
Writes colorized depth frames to stdout as: struct '<II' (jpeg_len, center_mm) + JPEG bytes.
Human-readable status goes to stderr as lines starting with "STATUS:".

Exit codes: 0 clean, 3 no depth device on the bus, 4 device present but could not be opened.
"""
import os
import signal
import struct
import sys

import cv2
import numpy as np

LIBS = os.environ.get("ORBBEC_OPENNI_LIBS", "/home/asnano/openni_orbbec/libs")
NEAR_MM, FAR_MM = 250, 1500  # colormap range
OUT_EVERY = 2  # sensor runs at 30 fps; send every 2nd frame (15 fps)
FRAME_OUT = None


def status(msg):
    try:
        print(f"STATUS: {msg}", file=sys.stderr, flush=True)
    except (BrokenPipeError, OSError):
        hard_exit(1)  # parent is gone


def die_with_parent():
    """Exit if the web app dies, so an orphaned worker never holds the sensor."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    parent = os.environ.get("SO101_PARENT_PID")
    if parent and os.getppid() != int(parent):
        hard_exit(0)


def err_text(e):
    """OpenNIError.__str__ raises on Python 3.12, so dig the message out of args."""
    try:
        msg = e.args[1] if len(e.args) > 1 else e.args[0]
        return msg.decode(errors="replace") if isinstance(msg, bytes) else str(msg)
    except Exception:
        return type(e).__name__


def hard_exit(code):
    # openni2.unload() double-frees in this driver build, so skip interpreter teardown.
    sys.stdout.flush()
    os._exit(code)


def colorize(depth):
    valid = depth > 0
    norm = np.clip((depth.astype(np.float32) - NEAR_MM) / (FAR_MM - NEAR_MM), 0, 1)
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def main():
    die_with_parent()
    # OpenNI's console log writes to fd 1, which carries our binary frames: keep a private copy of
    # the real stdout for frames and point fd 1 at stderr so driver logging becomes harmless text.
    global FRAME_OUT
    FRAME_OUT = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)

    from openni import openni2
    import openni_fix  # noqa: F401  (Orbbec OniDeviceInfo is larger than the wrapper's struct)

    try:
        openni2.initialize(LIBS)
    except Exception as e:
        status(f"OpenNI init failed: {err_text(e)}")
        hard_exit(4)

    if not openni2.Device.enumerate_uris():
        status("no depth device on the USB bus (replug the Astra)")
        hard_exit(3)

    try:
        dev = openni2.Device(None, mode=b"R")  # 'R': skip the driver's soft reset (SetMode 3) on open
        stream = dev.create_depth_stream()
        modes = stream.get_sensor_info().videoModes
        # Prefer 320x240 to keep USB 2.0 bandwidth free for the other cameras.
        pick = next((m for m in modes if m.resolutionX == 320 and m.resolutionY == 240
                     and m.pixelFormat == openni2.PIXEL_FORMAT_DEPTH_1_MM), None)
        if pick:
            stream.set_video_mode(pick)
        stream.start()
    except Exception as e:
        status(f"could not open depth device: {err_text(e)}")
        hard_exit(4)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    out = FRAME_OUT
    n = misses = 0
    try:
        while running:
            # read_frame() blocks forever if the sensor is unplugged or stalls; poll with a timeout instead.
            if openni2.wait_for_any_stream([stream], 1.0) is None:
                misses += 1
                if misses >= 5:
                    status("no depth frames for 5 s (sensor unplugged or stalled)")
                    hard_exit(4)  # skip stream.stop()/dev.close(): they can block after a disconnect
                continue
            misses = 0
            frame = stream.read_frame()
            n += 1
            if n == 1:
                status("ok")
            if n % OUT_EVERY:
                continue
            d = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(frame.height, frame.width)
            cy, cx = frame.height // 2, frame.width // 2
            patch = d[cy - 4:cy + 5, cx - 4:cx + 5]
            patch = patch[patch > 0]
            center = int(np.median(patch)) if patch.size else 0
            vis = colorize(d)
            cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                out.write(struct.pack("<II", len(jpg), center) + jpg.tobytes())
                out.flush()
    except (BrokenPipeError, ValueError):
        pass
    except Exception as e:
        status(f"depth stream error: {err_text(e)}")
        try:
            stream.stop()
        except Exception:
            pass
        hard_exit(4)
    try:
        stream.stop()
        dev.close()
    except Exception:
        pass
    hard_exit(0)


if __name__ == "__main__":
    main()
