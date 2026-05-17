"""End-to-end test for the UDP video handshake.

The GUI's ``UdpVideoClient`` publishes a ``VideoSubscribe`` message every
5 s with its host+port. The robot's ``UdpVideoStreamerThread`` subscribes
to that topic and latches the destination. This test runs both ends in
the same process and verifies the streamer's target gets set correctly
(and gets cleared on timeout).
"""

from __future__ import annotations

import time

import pytest

from src.core.bus.shim import messageHandlerSender
from src.core.bus.topics import VIDEO_SUBSCRIBE
from src.core.messaging.buffers import LatestFrameBuffer
from src.hardware.camera.threads.threadVideoStreamer import UdpVideoStreamerThread


def _make_sender():
    return messageHandlerSender({}, VIDEO_SUBSCRIBE)


def _publish_subscribe(sender, host: str, port: int) -> None:
    payload = {"host": host, "port": port, "fps": 15, "quality": 70}
    sender.send(payload)


def _wait_for_target(streamer, expected, sender, host, port, timeout_s=2.5):
    """Republish-then-drain in a tight loop until target matches or timeout.

    ZMQ pub/sub is slow-joiner — the first few publishes can be silently
    dropped while the SUB filter propagates. We loop publish+drain to
    survive that window.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _publish_subscribe(sender, host, port)
        time.sleep(0.05)
        streamer._drain_subscribe_messages()
        if streamer._get_target() == expected:
            return True
        time.sleep(0.02)
    return False


def test_streamer_latches_target_from_videosubscribe(broker):
    """A VideoSubscribe should populate ``_target`` after a few publish+drain
    cycles (slow-joiner tolerated by the publish retry loop).
    """
    buf = LatestFrameBuffer()
    streamer = UdpVideoStreamerThread(frame_buffer=buf)
    time.sleep(0.15)  # let the SUB filter reach the proxy
    sender = _make_sender()
    sender.send({"host": "warmup", "port": 1, "fps": 1, "quality": 1})  # warmup
    time.sleep(0.1)
    streamer._drain_subscribe_messages()
    got = _wait_for_target(streamer, ("203.0.113.42", 9999), sender,
                           "203.0.113.42", 9999)
    target = streamer._get_target()
    streamer._subscribe_sub.unsubscribe()
    assert got, f"target never latched (last={target})"
    assert target == ("203.0.113.42", 9999)


def test_streamer_clears_target_after_timeout(broker, monkeypatch):
    """If no VideoSubscribe arrives within SUBSCRIBE_TIMEOUT_S, the streamer
    drops the latched target (so a crashed GUI stops the encoder).
    """
    monkeypatch.setattr(UdpVideoStreamerThread, "SUBSCRIBE_TIMEOUT_S", 0.2)
    buf = LatestFrameBuffer()
    streamer = UdpVideoStreamerThread(frame_buffer=buf)
    time.sleep(0.15)
    sender = _make_sender()
    sender.send({"host": "warmup", "port": 1, "fps": 1, "quality": 1})
    time.sleep(0.1)
    streamer._drain_subscribe_messages()

    got = _wait_for_target(streamer, ("203.0.113.42", 9999), sender,
                           "203.0.113.42", 9999)
    assert got, "target never latched"

    # Now wait past the timeout WITHOUT republishing — the streamer must
    # drop the target on the next ``_check_subscribe_timeout`` call.
    time.sleep(0.35)
    streamer._check_subscribe_timeout()

    target = streamer._get_target()
    streamer._subscribe_sub.unsubscribe()
    assert target is None


def test_env_target_survives_timeout(broker, monkeypatch):
    """When a manual ``URT_VIDEO_TARGET_HOST/PORT`` is configured (headless
    test rig), the subscribe timeout falls back to it instead of clearing.
    """
    monkeypatch.setenv("URT_VIDEO_TARGET_HOST", "10.0.0.1")
    monkeypatch.setenv("URT_VIDEO_TARGET_PORT", "1234")
    monkeypatch.setattr(UdpVideoStreamerThread, "SUBSCRIBE_TIMEOUT_S", 0.2)
    buf = LatestFrameBuffer()
    streamer = UdpVideoStreamerThread(frame_buffer=buf)
    time.sleep(0.15)
    assert streamer._get_target() == ("10.0.0.1", 1234)

    sender = _make_sender()
    sender.send({"host": "warmup", "port": 1, "fps": 1, "quality": 1})
    time.sleep(0.1)
    streamer._drain_subscribe_messages()

    got = _wait_for_target(streamer, ("203.0.113.42", 9999), sender,
                           "203.0.113.42", 9999)
    assert got, "GUI announce never overrode env target"

    time.sleep(0.35)
    streamer._check_subscribe_timeout()
    target = streamer._get_target()
    streamer._subscribe_sub.unsubscribe()
    assert target == ("10.0.0.1", 1234), "env target should survive timeout"
