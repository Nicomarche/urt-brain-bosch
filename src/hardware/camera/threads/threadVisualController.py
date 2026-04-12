from __future__ import annotations

from src.hardware.camera.threads.threadLineFollowing import threadLineFollowing


class threadVisualController(threadLineFollowing):
    """Compatibility wrapper that runs line-following as a visual control producer."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("emit_motor_commands", False)
        kwargs.setdefault("tracking_state_direct_writes", False)
        super().__init__(*args, **kwargs)
