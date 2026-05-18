"""LocalizationConfig — dataclass agrupando los parámetros de localization.

Plan A1.2: el ``pose_estimator_thread.py`` (1893 LOC) tiene ~25
``getattr(_cfg_auto, ...)`` repartidos. Acá los agrupamos en un único
dataclass frozen leído una vez al boot. Mantiene las expresiones de
``config.py`` (math.pi, etc.) sin migrar a YAML rígido.

Razones para NO migrar a YAML:
  * ``config.py`` permite expresiones (``math.pi/2.0``); YAML no.
  * El ``ekf.yaml`` del REF es 80% ROS-specific.
  * Sintaxis Python da mejor diff en code review que escalares en YAML.

Para parámetros tuneables en campo (no físicos), usar
``runtime.yaml`` + ``RuntimeParams`` (plan E1).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalizationConfig:
    """Constantes inmutables de localization, leídas al boot.

    Si querés exponer un valor nuevo al pose_estimator, agregalo acá Y a
    ``load_localization_config()`` — un único lugar para que el orchestrator
    se entere.
    """

    # ── Dead reckoning ──────────────────────────────────────────────
    wheelbase_m: float
    rear_axle_to_cg_m: float
    max_integration_dt: float
    max_physical_yaw_rate_rads: float
    steer_gain_dr: float
    steer_lag_alpha: float
    encoder_curve_speed_compensation: bool

    # ── Camera lateral correction ───────────────────────────────────
    camera_correction_gain: float
    camera_correction_max_m: float
    camera_correction_step_max_m: float
    camera_correction_cooldown_s: float

    # ── Visual lane relocalization ──────────────────────────────────
    visual_lane_relocalization_speed_min_mps: float

    # ── Stopline visual relocalization ──────────────────────────────
    visual_stopline_event_max_age_s: float
    visual_stopline_max_map_error_m: float
    visual_stopline_relocalization_cooldown_s: float

    # ── Semantic relocalization ─────────────────────────────────────
    semantic_relocalization_cooldown_s: float
    semantic_relocalization_distance_tolerance_m: float
    semantic_relocalization_max_distance_m: float
    semantic_relocalization_max_map_error_m: float

    # ── Yaw EKF 1D ──────────────────────────────────────────────────
    yaw_ekf_q: float
    yaw_ekf_p_init: float
    yaw_ekf_r_steer_k: float
    yaw_ekf_r_straight: float
    imu_yaw_sign: int


def load_localization_config() -> LocalizationConfig:
    """Construye un ``LocalizationConfig`` a partir de las constantes
    legacy del módulo ``relocalization_thread``.

    Esto centraliza el lookup pero NO duplica la fuente de verdad —
    los valores siguen viniendo del legacy hasta A4, cuando se mudan
    a este módulo en la nueva arquitectura.
    """
    from src.localization import relocalization_thread as _legacy

    return LocalizationConfig(
        wheelbase_m=float(_legacy._WHEELBASE_M),
        rear_axle_to_cg_m=float(_legacy._REAR_AXLE_TO_CG_M),
        max_integration_dt=float(_legacy._MAX_INTEGRATION_DT),
        max_physical_yaw_rate_rads=float(_legacy._MAX_PHYSICAL_YAW_RATE_RADS),
        steer_gain_dr=float(_legacy._STEER_GAIN_DR),
        steer_lag_alpha=float(_legacy._STEER_LAG_ALPHA),
        encoder_curve_speed_compensation=bool(_legacy._ENCODER_CURVE_SPEED_COMPENSATION),
        camera_correction_gain=float(_legacy._CAMERA_LATERAL_CORRECTION_GAIN),
        camera_correction_max_m=float(_legacy._CAMERA_LATERAL_CORRECTION_MAX_M),
        camera_correction_step_max_m=float(_legacy._CAMERA_LATERAL_CORRECTION_STEP_MAX_M),
        camera_correction_cooldown_s=float(_legacy._CAMERA_LATERAL_CORRECTION_COOLDOWN_S),
        visual_lane_relocalization_speed_min_mps=float(
            _legacy._VISUAL_LANE_RELOCALIZATION_SPEED_MIN_MPS
        ),
        visual_stopline_event_max_age_s=float(_legacy._VISUAL_STOPLINE_EVENT_MAX_AGE_S),
        visual_stopline_max_map_error_m=float(_legacy._VISUAL_STOPLINE_MAX_MAP_ERROR_M),
        visual_stopline_relocalization_cooldown_s=float(
            _legacy._VISUAL_STOPLINE_RELOCALIZATION_COOLDOWN_S
        ),
        semantic_relocalization_cooldown_s=float(_legacy._SEMANTIC_RELOCALIZATION_COOLDOWN_S),
        semantic_relocalization_distance_tolerance_m=float(
            _legacy._SEMANTIC_RELOCALIZATION_DISTANCE_TOLERANCE_M
        ),
        semantic_relocalization_max_distance_m=float(
            _legacy._SEMANTIC_RELOCALIZATION_MAX_DISTANCE_M
        ),
        semantic_relocalization_max_map_error_m=float(
            _legacy._SEMANTIC_RELOCALIZATION_MAX_MAP_ERROR_M
        ),
        yaw_ekf_q=float(_legacy._YAW_EKF_Q),
        yaw_ekf_p_init=float(_legacy._YAW_EKF_P_INIT),
        yaw_ekf_r_steer_k=float(_legacy._YAW_EKF_R_STEER_K),
        yaw_ekf_r_straight=float(_legacy._YAW_EKF_R_STRAIGHT),
        imu_yaw_sign=int(_legacy._IMU_YAW_SIGN),
    )


__all__ = ["LocalizationConfig", "load_localization_config"]
