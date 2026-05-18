# src/behavior/visual_dr_blender.py
#
# Blender continuo entre el path visual (cámara → `LaneObservation.center_waypoints_body`)
# y el path de dead-reckoning (`RouteContext.route_waypoints` / lanelet centerline).
#
# Motivación:
#   El `VisualPrimaryGate` clásico decide binariamente: visual o DR. En la
#   transición 2-líneas → 1-línea → 0-líneas (entrada y salida de curvas
#   cerradas, sombras, parches del piso) ese switch produce saltos en el
#   `target_path` que se traducen en steer abrupto. La pista pide algo más
#   gradual: cuando veo las 2 líneas confío en visual, cuando veo 1 me
#   "guío más por DR pero sigo corrigiendo con la línea visible", y cuando
#   no veo nada me guío 100% por DR hasta volver a engancharme.
#
# Diseño:
#   - Reusamos `VisualPrimaryGate` para evaluar todos los hard-blocks
#     (map authority, scenario_not_allowed, geometría inválida, etc.). Si
#     el gate marca hard_block ⇒ α=0 (DR puro). Es la MISMA invariante
#     de seguridad que ya tenemos hoy — el blender no debilita ninguna.
#   - El α target se define por una tabla (2 líneas, 1 línea, 0 líneas)
#     con atenuación por `quality`. Después se suaviza hacia α target con
#     constantes de tiempo distintas para subir (recovery) y bajar (decay).
#   - Cuando `LANE_BLEND_ENABLED=False`, el blender se cortocircuita y
#     emite α ∈ {0, 1} según el gate clásico — comportamiento idéntico al
#     pre-blender (rollback gratis si el flag está apagado).
#
# Notas:
#   - El cálculo de α NO arma paths; eso lo hace `BaseScenario` (que tiene
#     pose, lanelet_map y demás contexto). El blender es un decisor puro.
#   - Stateful: guarda α_prev y last_tick_ts para integrar la suavización.

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.behavior.context import PlanningContext
from src.behavior.visual_primary_gate import VisualPrimaryGate, VisualPrimaryDecision
from src.core.types.perception import lane_observation_has_visual_path


@dataclass(frozen=True)
class BlendDecision:
    """Resultado de `VisualDrBlender.decide`.

    Args:
      alpha: peso ∈ [0, 1] del path visual. 1 = visual puro, 0 = DR puro.
      target_alpha: α* objetivo antes de suavizado temporal (útil para
        debug/tuning).
      num_lines: cantidad de líneas detectadas (0, 1, 2), derivada de
        `lane_observation.detected_sides`.
      visual_available: True si los waypoints visuales son utilizables
        como path (`lane_observation_has_visual_path` con quality mínima
        relajada — la calidad se aplica encima como atenuación).
      route_available: True si hay corredor DR (route_waypoints o lanelet)
        con el que mezclar. Cuando es False y visual_available=True el
        blender fuerza α=1 (no hay con qué mezclar).
      gate_decision: la decisión del `VisualPrimaryGate` underlying.
        Útil para mantener compatibilidad con notes del path.
      reason: motivo human-readable del α emitido.
      map_authority_active: refleja el flag del gate (intersection etc.).
      notes: dict para mergear con los notes del BehaviorOutput.
    """

    alpha: float
    target_alpha: float
    num_lines: int
    visual_available: bool
    route_available: bool
    gate_decision: VisualPrimaryDecision
    reason: str
    map_authority_active: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _BlenderConfig:
    enabled: bool
    alpha_two_line: float
    alpha_single_line: float
    alpha_no_line: float
    quality_min: float
    tau_decay_s: float
    tau_recovery_s: float
    min_points: int


class VisualDrBlender:
    """Decisor stateful que emite el peso α del path visual."""

    def __init__(self) -> None:
        self._gate = VisualPrimaryGate()
        self._alpha_prev: float = 0.0
        self._last_tick_s: float | None = None
        self._last_num_lines: int = 0

    @property
    def gate(self) -> VisualPrimaryGate:
        """Acceso al gate underlying (lo necesita BaseScenario para reusar el feature
        flag clásico cuando `LANE_BLEND_ENABLED=False`)."""
        return self._gate

    def decide(
        self,
        *,
        ctx: PlanningContext,
        lane_observation,
        scenario_name: str,
        route_corridor_available: bool,
        lanelet_corridor_available: bool,
        now_s: float | None = None,
    ) -> BlendDecision:
        cfg = _load_config()

        # El gate clásico nos da: hard_block (map authority / scenario / geometría),
        # la quality threshold por measurement_mode, y la hysteresis de entrada.
        # Reusamos su resultado para mantener invariantes de seguridad.
        gate_decision = self._gate.decide(
            ctx=ctx,
            lane_observation=lane_observation,
            scenario_name=scenario_name,
            route_corridor_available=route_corridor_available,
            lanelet_corridor_available=lanelet_corridor_available,
        )

        route_available = bool(route_corridor_available or lanelet_corridor_available)
        num_lines = _count_lines(lane_observation)
        self._last_num_lines = num_lines
        # `visual_available` mira los waypoints crudos con quality MUY relajada:
        # el blender corrige la influencia con la curva de α(quality) más abajo.
        # No queremos que un frame con quality 0.6 nos saque a 100% DR de golpe.
        visual_available = bool(
            lane_observation is not None
            and lane_observation_has_visual_path(
                lane_observation,
                min_quality=0.0,
                min_points=cfg.min_points,
            )
        )

        notes: dict[str, Any] = dict(gate_decision.notes)
        notes.update({
            "lane_blend_num_lines": int(num_lines),
            "lane_blend_visual_available": bool(visual_available),
            "lane_blend_dr_available": bool(route_available),
        })

        # ── Modo legacy: feature flag apagado → comportamiento binario clásico ──
        if not cfg.enabled:
            legacy_alpha = 1.0 if bool(gate_decision.use_visual_path) else 0.0
            self._alpha_prev = legacy_alpha
            self._last_tick_s = float(now_s) if now_s is not None else self._last_tick_s
            notes.update({
                "lane_blend_enabled": False,
                "lane_blend_alpha": legacy_alpha,
                "lane_blend_target_alpha": legacy_alpha,
            })
            return BlendDecision(
                alpha=legacy_alpha,
                target_alpha=legacy_alpha,
                num_lines=num_lines,
                visual_available=visual_available,
                route_available=route_available,
                gate_decision=gate_decision,
                reason=f"blend_disabled:{gate_decision.reason}",
                map_authority_active=bool(gate_decision.map_authority_active),
                notes=notes,
            )

        # ── α* (target) ──────────────────────────────────────────────────────
        # Hard-block del gate: respetamos. Map authority y geometría inválida
        # nos mandan a 0 inmediatamente (instantáneo, sin suavizado).
        hard_block = bool(gate_decision.notes.get("visual_primary_hard_block", False))
        if hard_block:
            target_alpha = 0.0
            reason = f"hard_block:{gate_decision.reason}"
            instant = True
        elif not visual_available:
            target_alpha = 0.0
            reason = "visual_unavailable"
            instant = True
        elif not route_available:
            # Sin corredor DR no hay con qué mezclar — mantener visual primary
            # como hoy (reusa el resultado del gate clásico).
            target_alpha = 1.0 if bool(gate_decision.use_visual_path) else 0.0
            reason = f"no_route_corridor:gate:{gate_decision.reason}"
            instant = True
        else:
            target_alpha, reason = _target_alpha_for_lines(num_lines, lane_observation, cfg)
            instant = False

        target_alpha = float(_clamp01(target_alpha))

        # ── Suavizado temporal ──────────────────────────────────────────────
        if instant:
            alpha_new = target_alpha
            self._alpha_prev = alpha_new
            self._last_tick_s = float(now_s) if now_s is not None else self._last_tick_s
        else:
            alpha_new = self._smooth_toward(target_alpha, cfg=cfg, now_s=now_s)

        notes.update({
            "lane_blend_enabled": True,
            "lane_blend_alpha": float(alpha_new),
            "lane_blend_target_alpha": float(target_alpha),
            "lane_blend_reason": str(reason),
        })

        return BlendDecision(
            alpha=float(alpha_new),
            target_alpha=float(target_alpha),
            num_lines=num_lines,
            visual_available=visual_available,
            route_available=route_available,
            gate_decision=gate_decision,
            reason=str(reason),
            map_authority_active=bool(gate_decision.map_authority_active),
            notes=notes,
        )

    def _smooth_toward(
        self,
        target_alpha: float,
        *,
        cfg: _BlenderConfig,
        now_s: float | None,
    ) -> float:
        current_ts = float(now_s) if now_s is not None else time.time()
        last_ts = self._last_tick_s if self._last_tick_s is not None else current_ts
        dt = max(0.0, current_ts - last_ts)
        self._last_tick_s = current_ts

        # Constantes de tiempo asimétricas: queremos REACCIONAR rápido al
        # perder visión (subir DR) y subir LENTO al recuperarla (evitar que
        # un solo frame bueno reactive visual y vuelva a saltar).
        going_up = target_alpha > self._alpha_prev
        tau = float(cfg.tau_recovery_s if going_up else cfg.tau_decay_s)
        if tau <= 1e-6 or dt <= 0.0:
            alpha_new = target_alpha
        else:
            # Filtro exponencial de primer orden discreto.
            blend = 1.0 - math.exp(-dt / tau)
            alpha_new = self._alpha_prev + blend * (target_alpha - self._alpha_prev)

        alpha_new = _clamp01(alpha_new)
        self._alpha_prev = alpha_new
        return alpha_new


def _count_lines(lane_observation) -> int:
    if lane_observation is None:
        return 0
    sides = tuple(getattr(lane_observation, "detected_sides", ()) or ())
    return int(min(2, len(sides)))


def _target_alpha_for_lines(
    num_lines: int,
    lane_observation,
    cfg: _BlenderConfig,
) -> tuple[float, str]:
    """α* base por conteo de líneas, atenuado por quality."""
    if num_lines >= 2:
        base = cfg.alpha_two_line
        reason = "two_line"
    elif num_lines == 1:
        base = cfg.alpha_single_line
        reason = "single_line"
    else:
        base = cfg.alpha_no_line
        reason = "no_line"

    quality_factor = _quality_attenuation(
        float(getattr(lane_observation, "quality", 0.0) or 0.0),
        quality_min=cfg.quality_min,
    )
    return float(base) * float(quality_factor), reason


def _quality_attenuation(quality: float, *, quality_min: float) -> float:
    """Atenuador lineal: por debajo de `quality_min` se atenúa a 0, arriba sube
    linealmente hasta 1.

    quality=quality_min ⇒ 0
    quality=1.0         ⇒ 1
    Cualquier valor entre esos dos se interpola lineal.
    """
    quality = float(quality)
    quality_min = float(quality_min)
    if quality <= quality_min:
        return 0.0
    if quality >= 1.0:
        return 1.0
    return (quality - quality_min) / max(1.0 - quality_min, 1e-6)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_config() -> _BlenderConfig:
    try:
        import config as _cfg
    except Exception:
        _cfg = None

    def _get(name: str, default):
        return getattr(_cfg, name, default) if _cfg is not None else default

    return _BlenderConfig(
        enabled=bool(_get("LANE_BLEND_ENABLED", False)),
        alpha_two_line=float(_get("LANE_BLEND_ALPHA_TWO_LINE", 1.0)),
        alpha_single_line=float(_get("LANE_BLEND_ALPHA_SINGLE_LINE", 0.3)),
        alpha_no_line=float(_get("LANE_BLEND_ALPHA_NO_LINE", 0.0)),
        quality_min=float(_get("LANE_BLEND_QUALITY_MIN", 0.5)),
        tau_decay_s=max(0.0, float(_get("LANE_BLEND_TAU_DECAY_S", 0.3))),
        tau_recovery_s=max(0.0, float(_get("LANE_BLEND_TAU_RECOVERY_S", 0.5))),
        min_points=int(_get("LANE_VISUAL_MIN_POLY_POINTS", 8)),
    )
