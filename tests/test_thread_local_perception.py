from __future__ import annotations

from src.hardware.camera.threads.threadLocalPerception import threadLocalPerception


class _RecordingSender:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(str(message))


def test_walk_area_crossed_does_not_request_parking_mode() -> None:
    thread = threadLocalPerception.__new__(threadLocalPerception)
    thread._current_mode = "auto"
    thread._walk_area_stop_duration = 3.0
    thread._walk_area_min_box_area = 0.04
    thread._walk_area_active = False
    thread._walk_area_no_obstacle_since = None
    thread._walk_area_cooldown = 10.0
    thread._walk_area_last_cleared = 0.0
    thread.stateChangeSender = _RecordingSender()

    detections = [{"class": "walk_area", "box": [0.0, 0.0, 0.5, 0.5]}]

    thread._handle_walk_area(detections, now=20.0)
    assert thread._walk_area_active is True
    assert thread.stateChangeSender.messages == []

    thread._handle_walk_area(detections, now=23.5)

    assert thread._walk_area_active is False
    assert thread._walk_area_no_obstacle_since is None
    assert thread._walk_area_last_cleared == 23.5
    assert thread.stateChangeSender.messages == []
