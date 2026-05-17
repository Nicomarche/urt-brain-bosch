"""Shared QGraphicsView subclass with zoom + pan controls.

Used by both ``map_view`` (operator) and ``map_editor`` (authoring) so the
two surfaces feel identical to navigate. Behaviour:

* **Mouse wheel** (or Ctrl+wheel): zoom around the cursor.
* **Trackpad two-finger swipe**: pan. We detect this by ``pixelDelta() != 0``
  on Mac/Wayland — Qt synthesises wheel events from the OS gesture, and
  pixelDelta is only set when the input device is a precision pointer.
* **Right-button drag** or **middle-button drag**: pan (the mouse-only
  fallback for users without a trackpad).
* **Pinch gesture** (Mac trackpad): zoom around the gesture centre. This
  is delivered as a ``QPinchGesture`` event, separate from the wheel.
* **Left button** is never grabbed — clicks pass through to the scene so
  the map can do its own hit-testing (add node / select / relocate).

Why not ``QGraphicsView.ScrollHandDrag``?
   It hijacks the left button for panning, which conflicts with the
   click-to-add-node and click-to-relocate flows we need. Owning the
   right/middle button for pan keeps the left button free.

Zoom is clamped so the user can't accidentally lose the scene. The
transform anchor is ``AnchorUnderMouse`` so a wheel-zoom keeps the point
under the cursor stable, which is what every map app does.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtGui import QMouseEvent, QPainter, QWheelEvent
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView


class PannableZoomableView(QGraphicsView):
    """``QGraphicsView`` with wheel-zoom, trackpad-pan and right-drag pan.

    Drop-in replacement: instantiate with ``(scene, parent)`` exactly like
    the base class. The owning widget keeps full control over what to do
    with left-clicks (the scene's mouse events still fire normally).
    """

    # How many "zoom-units" per pixel of vertical wheel angle.
    # 0.0015 → ~10% per notch on a typical mouse wheel (120 units).
    WHEEL_ZOOM_GAIN = 0.0015
    # Hard limits keep the scene reachable.
    MIN_ZOOM = 0.05
    MAX_ZOOM = 30.0

    def __init__(self, scene: QGraphicsScene,
                 parent: Optional["QGraphicsView"] = None):
        super().__init__(scene, parent)

        # Smooth rendering for the rasterised SVG / antialiased lines.
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Wheel-zoom feels right when the point under the cursor stays put.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Resizes don't yank the view — keep scene-fixed point under cursor.
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # We do panning ourselves on right/middle button — turn off the
        # built-in left-button ScrollHandDrag so left clicks reach the scene.
        self.setDragMode(QGraphicsView.NoDrag)

        # Mac trackpad: opt into pinch gestures so we get QGestureEvents.
        # Two-finger swipe still arrives as wheel events with pixelDelta.
        self.viewport().grabGesture(Qt.PinchGesture)

        # Pan state — populated on right-button press.
        self._panning = False
        self._pan_last = QPoint()

    # ------------------------------------------------------------------
    # Wheel: trackpad two-finger swipe → pan, vertical wheel → zoom.
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt
        # ``pixelDelta()`` is non-zero only for high-resolution / trackpad
        # devices. A Logitech wheel reports angleDelta=±120 and pixelDelta=0,
        # so this neatly splits "physical scroll wheel" (zoom) from
        # "two-finger trackpad swipe" (pan) without modifier keys. Holding
        # Ctrl/Cmd forces zoom even on the trackpad — handy on a Mac.
        pixel = event.pixelDelta()
        modifiers = event.modifiers()
        forces_zoom = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))

        if not pixel.isNull() and not forces_zoom:
            # Trackpad pan: shift the scrollbars by the raw pixel delta.
            self._pan_by(pixel.x(), pixel.y())
            event.accept()
            return

        # Zoom: prefer angleDelta (works on every device), fall back to
        # pixelDelta when only that is available.
        delta = event.angleDelta().y()
        if delta == 0 and not pixel.isNull():
            delta = pixel.y() * 8   # rough conversion to "1/8°"
        if delta == 0:
            super().wheelEvent(event)
            return

        factor = 1.0 + self.WHEEL_ZOOM_GAIN * delta
        self._apply_zoom(factor)
        event.accept()

    # ------------------------------------------------------------------
    # Right / middle button drag = pan.
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = True
            self._pan_last = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self._pan_by(delta.x(), delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning and event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Pinch gesture (Mac trackpad).
    # ------------------------------------------------------------------
    def event(self, event):  # noqa: N802 - Qt
        if event.type() == QEvent.Gesture:
            pinch = event.gesture(Qt.PinchGesture)
            if pinch is not None:
                # Native scaleFactor is the *incremental* multiplier since
                # the last gesture update — apply directly.
                self._apply_zoom(pinch.scaleFactor())
                event.accept()
                return True
        return super().event(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pan_by(self, dx: int, dy: int) -> None:
        """Shift the view by (dx, dy) viewport pixels."""
        h = self.horizontalScrollBar()
        v = self.verticalScrollBar()
        h.setValue(h.value() - dx)
        v.setValue(v.value() - dy)

    def current_zoom(self) -> float:
        """Current scale factor (m11 of the view transform). 1.0 = identity."""
        return float(self.transform().m11())

    def _apply_zoom(self, factor: float) -> None:
        """Multiply the current zoom by ``factor`` if within bounds."""
        if factor <= 0:
            return
        new_zoom = self.current_zoom() * factor
        if new_zoom < self.MIN_ZOOM:
            factor = self.MIN_ZOOM / self.current_zoom()
        elif new_zoom > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / self.current_zoom()
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        """Reset to identity — useful after a "Fit view"."""
        self.resetTransform()
