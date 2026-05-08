#!/usr/bin/env python3
"""External viewer for line-following debug previews (opt-in).

The default operator surface is the unified PyQt5 GUI (``src/dashboard/gui``),
which subscribes to ``LineFollowingDebug`` over SocketIO and shows a single
composite frame per tick. That covers 95% of debugging needs.

This script is the *advanced* path: when you want to see the individual
processing stages (binary mask, ROI, etc.) at full rate, run the line
following thread with ``URT_PREVIEW_BACKEND=zmq`` and launch this viewer
separately. It keeps cv2.imshow out of the worker process (which would
crash on macOS forks and lock the loop on the Jetson) by hosting the
windows in a clean process tree of its own.

Usage:
    URT_PREVIEW_BACKEND=zmq ./run.sh
    # in another terminal:
    python3.13 view_debug.py                      # subscribe to all windows
    python3.13 view_debug.py --endpoint tcp://localhost:5577
    python3.13 view_debug.py --topics "1. Final Result,Binary"
Press q on any window to quit.
"""
import argparse
import sys
import time

import cv2
import numpy as np
import zmq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="tcp://localhost:5577")
    p.add_argument("--topics", default="",
                   help="Comma-separated window names to subscribe to (default: all)")
    args = p.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.endpoint)
    if args.topics:
        for t in args.topics.split(","):
            sock.subscribe(t.strip().encode("utf-8"))
    else:
        sock.subscribe(b"")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    print(f"[view_debug] subscribed {args.endpoint} (topics={args.topics or 'all'}) — q to quit")

    last_log = time.time()
    counts = {}
    try:
        while True:
            socks = dict(poller.poll(timeout=50))
            if sock in socks:
                while True:
                    try:
                        parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if len(parts) < 2:
                        continue
                    name = parts[0].decode("utf-8", errors="replace")
                    arr = np.frombuffer(parts[1], dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    cv2.imshow(name, img)
                    counts[name] = counts.get(name, 0) + 1
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            now = time.time()
            if now - last_log >= 5.0 and counts:
                rates = ", ".join(f"{k}={v / (now - last_log):.1f}fps" for k, v in counts.items())
                print(f"[view_debug] {rates}")
                counts.clear()
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
