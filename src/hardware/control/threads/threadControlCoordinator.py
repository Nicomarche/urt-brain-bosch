from __future__ import annotations

import time

import config as _config
from src.hardware.control.navigationControlPolicy import NavigationControlPolicy
from src.core.types import ControlDecision, VisualControlCandidate
from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import SpeedMotor, SteerMotor, StateChange
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber


class threadControlCoordinator(ThreadWithStop):
    """Single-writer command coordinator for steering and speed messages."""

    def __init__(
        self,
        queuesList,
        tracking_state,
        visual_candidate_buffer,
        control_decision_buffer=None,
        *,
        sign_action_event=None,
        highway_mode_event=None,
        steer_override_event=None,
    ):
        super().__init__(pause=0.05)
        self.queuesList = queuesList
        self.tracking_state = tracking_state
        self.visual_candidate_buffer = visual_candidate_buffer
        self.control_decision_buffer = control_decision_buffer
        self.sign_action_event = sign_action_event
        self.highway_mode_event = highway_mode_event
        self.steer_override_event = steer_override_event
        self.policy = NavigationControlPolicy()
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.stateChangeSubscriber = messageHandlerSubscriber(self.queuesList, StateChange, "lastOnly", True)
        self._current_state = "DEFAULT"
        self._sign_action_was_active = False
        self._candidate_stale_timeout_s = float(
            getattr(_config, "CONTROL_COORDINATOR_CANDIDATE_STALE_TIMEOUT_S", 0.5) or 0.5
        )
        self._resume_min_speed = float(getattr(_config, "LF_MIN_SPEED", 8.0) or 8.0)
        self._resume_highway_min_speed = float(getattr(_config, "LF_HIGHWAY_MIN_SPEED", 15.0) or 15.0)

    def _consume_state_change(self) -> None:
        message = self.stateChangeSubscriber.receive()
        if message is None:
            return
        self._current_state = str(message or "").strip().upper() or "DEFAULT"

    def _effective_resume_speed(self) -> float:
        if self.highway_mode_event is not None and self.highway_mode_event.is_set():
            return float(self._resume_highway_min_speed)
        return float(self._resume_min_speed)

    def _send_decision(self, decision: ControlDecision) -> None:
        if decision.steer_sent and decision.steering_deg is not None:
            steer_value = int(round(float(decision.steering_deg) * 10.0))
            self.steerMotorSender.send(str(steer_value))
        if decision.speed_sent and decision.speed_cmd is not None:
            speed_value = int(round(float(decision.speed_cmd) * 10.0))
            self.speedMotorSender.send(str(speed_value))

    def thread_work(self):
        self._consume_state_change()
        candidate, candidate_ts, _ = self.visual_candidate_buffer.read_latest(with_metadata=True)
        now = time.time()

        if not isinstance(candidate, VisualControlCandidate):
            decision = self.policy.decide(None, self.tracking_state)
        else:
            candidate_age_s = max(0.0, now - float(candidate_ts or candidate.timestamp or now))
            if candidate_age_s > self._candidate_stale_timeout_s:
                decision = ControlDecision(
                    timestamp=now,
                    steering_deg=0.0,
                    speed_cmd=0.0,
                    authority="safety",
                    safety_override=True,
                    reason="stale_candidate",
                    speed_sent=False,
                    steer_sent=False,
                    source_candidate=str(candidate.source or "none"),
                )
            else:
                decision = self.policy.decide(
                    candidate,
                    self.tracking_state,
                    sign_action_active=bool(self.sign_action_event and self.sign_action_event.is_set()),
                    steer_override_active=bool(self.steer_override_event and self.steer_override_event.is_set()),
                )

        if self._current_state not in {"AUTO", "PARKING"}:
            decision = ControlDecision(
                timestamp=decision.timestamp,
                steering_deg=decision.steering_deg,
                speed_cmd=decision.speed_cmd,
                authority="inactive",
                safety_override=decision.safety_override,
                reason="line_following_inactive",
                speed_sent=False,
                steer_sent=False,
                source_candidate=decision.source_candidate,
                debug=dict(decision.debug or {}),
            )

        sign_active = bool(self.sign_action_event and self.sign_action_event.is_set())
        if self._sign_action_was_active and not sign_active and decision.speed_sent and decision.speed_cmd is not None:
            decision = ControlDecision(
                timestamp=decision.timestamp,
                steering_deg=decision.steering_deg,
                speed_cmd=self._effective_resume_speed(),
                authority=decision.authority,
                safety_override=decision.safety_override,
                reason=f"{decision.reason}:resume_after_sign",
                speed_sent=decision.speed_sent,
                steer_sent=decision.steer_sent,
                source_candidate=decision.source_candidate,
                debug=dict(decision.debug or {}),
            )
        self._sign_action_was_active = sign_active

        self._send_decision(decision)

        if self.control_decision_buffer is not None:
            self.control_decision_buffer.write(decision, timestamp=decision.timestamp)
        if self.tracking_state is not None and hasattr(self.tracking_state, "update_from_control_decision"):
            self.tracking_state.update_from_control_decision(decision)
