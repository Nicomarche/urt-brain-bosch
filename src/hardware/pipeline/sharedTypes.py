from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class LaneObservation:
    timestamp: float = 0.0
    source_mode: str = "unknown"
    detected_sides: tuple[str, ...] = field(default_factory=tuple)
    lateral_offset_m: float | None = None
    heading_error_rad: float = 0.0
    direct_error_m: float | None = None
    lane_width_px: float | None = None
    quality: float = 0.0
    curve_hint: str = "STRAIGHT"
    camera_yaw_hint_rad: float | None = None
    camera_yaw_hint_confidence: float = 0.0
    blind_mode: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoplineObservation:
    timestamp: float = 0.0
    visible: bool = False
    stable: bool = False
    distance_m: float | None = None
    confidence: float = 0.0
    pass_event: dict[str, Any] | None = None
    expected_node_id: str | None = None
    expected_node_attr: int = 0
    source: str = "none"
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseEstimate:
    timestamp: float = 0.0
    raw_pose: Pose2D = field(default_factory=Pose2D)
    fused_pose: Pose2D = field(default_factory=Pose2D)
    speed_mps: float = 0.0
    yaw_rad: float = 0.0
    steer_rad: float = 0.0
    speed_source: str = "none"
    speed_feedback_age_s: float | None = None
    speed_command_age_s: float | None = None
    localization_confidence: float = 0.0
    relocalization_mode: str = "dead_reckoning"
    last_relocalization_source: str = "none"
    last_relocalization_error_m: float = 0.0
    raw_lateral_error_m: float = 0.0
    lane_measurement_reliable: bool = False
    camera_lateral_correction_m: float = 0.0
    imu_received: bool = False


@dataclass(frozen=True)
class RouteContext:
    timestamp: float = 0.0
    route_active: bool = False
    route_id: str | None = None
    route_points: list[dict[str, float]] = field(default_factory=list)
    destination_node_id: str | None = None
    destination_label: str | None = None
    route_queue: list[dict[str, Any]] = field(default_factory=list)
    current_node_id: str | None = None
    current_node_attr: int = 0
    upcoming_node_id: str | None = None
    upcoming_node_attr: int = 0
    maneuver_type: str = "none"
    route_progress: float = 0.0
    route_completed: bool = False
    waypoint_mode_active: bool = False
    matched_idx: int = 0
    target_idx: int = 0
    matched_pose: Pose2D = field(default_factory=Pose2D)
    path_psi: float = 0.0
    path_kappa: float = 0.0
    path_heading_change_rad: float = 0.0
    error_m: float = 0.0
    heading_rad: float = 0.0
    map_match_error_m: float = 0.0
    remaining_distance_m: float = 0.0
    replans: int = 0
    route_source: str = "none"
    destination_point: dict[str, float] | None = None
    next_semantic_id: str | None = None
    next_semantic_type: str | None = None
    next_semantic_label: str | None = None
    next_semantic_distance_m: float | None = None
    expected_control_type: str | None = None
    current_zone_ids: list[str] = field(default_factory=list)
    current_zone_types: list[str] = field(default_factory=list)
    map_metadata: dict[str, Any] = field(default_factory=dict)
    available_destinations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class VisualControlCandidate:
    timestamp: float = 0.0
    steering_deg: float | None = None
    speed_cmd: float | None = None
    confidence: float = 0.0
    blind_mode: str | None = None
    source: str = "none"
    direct_error_m: float | None = None
    active: bool = False
    command_source: str = "none"
    computed_steering_deg: float | None = None
    computed_speed_cmd: float | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlDecision:
    timestamp: float = 0.0
    steering_deg: float | None = None
    speed_cmd: float | None = None
    authority: str = "none"
    safety_override: bool = False
    reason: str = "none"
    speed_sent: bool = False
    steer_sent: bool = False
    source_candidate: str = "none"
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualStateSnapshot:
    timestamp: float = 0.0
    frame_sequence: int = 0
    detection_mode: str = "unknown"
    active: bool = False
    curve_state: str = "STRAIGHT"
    heading_error_rad: float = 0.0
    camera_yaw_hint_rad: float | None = None
    camera_yaw_hint_confidence: float = 0.0
    frame_trace: dict[str, Any] = field(default_factory=dict)
    local_lane_payload: dict[str, Any] = field(default_factory=dict)
    stopline_debug: dict[str, Any] = field(default_factory=dict)
    candidate: VisualControlCandidate | None = None
