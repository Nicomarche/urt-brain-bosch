// line-following.component.ts
//
// Componente Angular del panel "Line Following" del dashboard de operador.
//
// Responsabilidad: ofrecer una UI mínima para:
//   1. Mostrar el estado del pipeline de percepción (AI Local YOLO TensorRT 416 px).
//   2. Configurar parámetros de stream de debug (FPS, calidad, escala).
//   3. Exponer sliders editables para velocidad de seguimiento de carril y
//      recuperación de curvas cerradas.
//   4. Reflejar el estado real (motor habilitado, comando aplicado, conexión serial).
//
// Por qué AI Local solamente: los modos legacy (OpenCV, LSTR, HybridNets remoto,
// Supercombo) fueron eliminados durante el cleanup de la Fase 0 del port a
// arquitectura "Autoware-light". Hoy la única fuente de verdad de detección de
// carriles es el modelo YOLO TensorRT que corre embebido en el proceso de cámara.
//
// Comunicación: este componente envía mensajes 'LineFollowingConfig' por
// WebSocket al backend Flask, y se suscribe a los streams 'lineFollowingDebug'
// (frame JPEG base64) y 'lineFollowingStatus' (telemetría JSON).
//
// Persistencia: la última configuración se guarda en localStorage para que el
// operador no pierda el setup entre recargas del navegador.

import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { WebSocketService } from '../../webSocket/web-socket.service';
import { Subscription } from 'rxjs';

interface SliderConfig {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  group: string;
}

interface DebugView {
  id: number;
  name: string;
  icon: string;
  description: string;
}

interface DebugStatus {
  steering: number | null;
  speed: number | null;
  mode: string;
  view: string;
  active: boolean;
  computed_steering?: number | null;
  computed_speed?: number | null;
  commanded_steering?: number | null;
  commanded_speed?: number | null;
  command_source?: string;
  actual_steering?: number | null;
  actual_speed?: number | null;
  serial_connected?: boolean | null;
  engine_enabled?: boolean | null;
  klem_mode?: number | null;
  actuator_command_type?: string | null;
  actuator_speed_x10?: number | null;
  actuator_steer_x10?: number | null;
  actuator_blocked_reason?: string | null;
  local_ai_ready?: boolean;
  local_ai_infer_ms?: number;
  local_ai_fps?: number;
}

@Component({
  selector: 'app-line-following',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './line-following.component.html',
  styleUrls: ['./line-following.component.css']
})
export class LineFollowingComponent implements OnInit, OnDestroy {

  // El único modo soportado tras el cleanup. Se mantiene la propiedad por compatibilidad
  // con el contrato de mensajes del backend ('detection_mode'), pero no es seleccionable.
  readonly selectedMode: string = 'ai_local';

  // Parámetros del runtime AI Local. min_confidence y input_size se leen una sola vez
  // al inicializar el detector — cambiarlos requiere reinicio del proceso de cámara.
  // 'interval' es el período de inferencia (s); subirlo baja FPS y CPU.
  localAiMinConfidence: number = 0.35;
  localAiInputSize: number = 320;
  localAiInterval: number = 0.04;
  localAiReady: boolean = false;
  localAiFps: number = 0;
  localAiInferMs: number = 0;

  // Selector de vista de debug. id 0 = stream apagado (ahorra ancho de banda).
  // El resto son frames intermedios del pipeline que el thread de cámara emite por WS.
  selectedDebugView: number = 0;
  debugViews: DebugView[] = [
    { id: 0, name: 'Apagado', icon: 'OFF', description: 'Stream de debug desactivado para máximo rendimiento.' },
    { id: 1, name: 'Final', icon: 'F', description: 'Resultado final con líneas detectadas y trayectoria calculada.' },
    { id: 8, name: 'Bird Eye', icon: 'BEV', description: 'Vista de pájaro (perspectiva superior).' },
    { id: 11, name: 'AI Local', icon: 'AI', description: 'Salida del modelo YOLO TensorRT con máscara de carriles.' },
  ];

  // Stream de debug — el operador puede bajar FPS/calidad/escala para conservar
  // ancho de banda en pista (4G) sin perder visibilidad del pipeline.
  streamFps: number = 5;
  streamQuality: number = 50;
  streamScale: number = 0.5;

  // Toggle de la rutina de recovery de curvas cerradas. Sigue siendo útil en AI Local
  // mientras la lógica de "salir de carril" no migre al BehaviorPlanner (Fase 4).
  useCurveRecovery: boolean = true;

  // Estado UI: qué tarjetas de configuración están desplegadas. Solo conservamos
  // los grupos vigentes en AI Local (speed) más recovery (transición a Fase 4).
  expandedSections: { [key: string]: boolean } = {
    speed: false,
    recovery: false,
  };

  // Sliders activos. El array se mantiene como lista plana porque el template lo
  // filtra por grupo vía getSlidersByGroup() — agregar un grupo nuevo no requiere
  // tocar la UI ni el sendConfig().
  sliders: SliderConfig[] = [
    { key: 'base_speed', label: 'Velocidad Base', min: 5, max: 40, step: 1, value: 15, group: 'speed' },
    { key: 'max_speed', label: 'Velocidad Máxima', min: 10, max: 50, step: 1, value: 25, group: 'speed' },
    { key: 'min_speed', label: 'Velocidad Mínima', min: 5, max: 30, step: 1, value: 8, group: 'speed' },
    { key: 'recovery_max_steer_frames', label: 'Frames Max Giro', min: 2, max: 15, step: 1, value: 6, group: 'recovery' },
    { key: 'recovery_reverse_speed', label: 'Velocidad Reversa', min: -20, max: -3, step: 1, value: -8, group: 'recovery' },
    { key: 'recovery_reverse_time_min', label: 'Reversa Min (s)', min: 0.1, max: 1.0, step: 0.05, value: 0.3, group: 'recovery' },
    { key: 'recovery_reverse_time_max', label: 'Reversa Max (s)', min: 0.5, max: 3.0, step: 0.1, value: 1.5, group: 'recovery' },
    { key: 'recovery_reverse_steer_scale', label: 'Escala Giro Reversa', min: 0.5, max: 3.0, step: 0.1, value: 1.5, group: 'recovery' },
    { key: 'recovery_pre_turn_time', label: 'Tiempo Pre-Giro (s)', min: 0.1, max: 1.5, step: 0.05, value: 0.6, group: 'recovery' },
    { key: 'recovery_realign_time', label: 'Tiempo Realinear (s)', min: 0.1, max: 1.0, step: 0.05, value: 0.6, group: 'recovery' },
    { key: 'recovery_error_shrink_ratio', label: 'Ratio Error Corrigiendo', min: 0.5, max: 0.95, step: 0.05, value: 0.85, group: 'recovery' },
  ];

  private updateTimeout: any = null;
  private debugStreamSubscription: Subscription | null = null;
  private debugStatusSubscription: Subscription | null = null;

  debugImageSrc: string | null = null;
  debugStatus: DebugStatus | null = null;
  debugStreamEnabled: boolean = true;

  get debugViewName(): string {
    const view = this.debugViews.find(v => v.id === this.selectedDebugView);
    return view ? view.name : 'Apagado';
  }

  constructor(private webSocketService: WebSocketService) {}

  ngOnInit(): void {
    // Restaurar última configuración del operador. Si el JSON está corrupto,
    // se ignora silenciosamente — no bloqueamos el panel por un localStorage roto.
    const savedConfig = localStorage.getItem('lineFollowingConfig');
    if (savedConfig) {
      try {
        const config = JSON.parse(savedConfig);
        this.applyConfig(config);
      } catch (e) {
        console.error('Failed to load saved config:', e);
      }
    }

    // Stream de imagen: el backend manda JPEGs base64 cada streamFps frames.
    // El template los muestra vía data URI directo en un <img>.
    this.debugStreamSubscription = this.webSocketService
      .receiveLineFollowingDebug()
      .subscribe((imageData: string) => {
        if (imageData) {
          this.debugImageSrc = 'data:image/jpeg;base64,' + imageData;
        }
      });

    // Stream de estado: telemetría a ~5 Hz con steering/speed comandados,
    // estado del actuador y métricas del modelo AI. Sirve para diagnóstico en pista.
    this.debugStatusSubscription = this.webSocketService
      .receiveLineFollowingStatus()
      .subscribe((status: DebugStatus) => {
        this.debugStatus = status;
        this.localAiReady = status?.local_ai_ready ?? false;
        this.localAiInferMs = status?.local_ai_infer_ms ?? 0;
        this.localAiFps = status?.local_ai_fps ?? 0;
      });
  }

  ngOnDestroy(): void {
    if (this.updateTimeout) {
      clearTimeout(this.updateTimeout);
    }
    if (this.debugStreamSubscription) {
      this.debugStreamSubscription.unsubscribe();
    }
    if (this.debugStatusSubscription) {
      this.debugStatusSubscription.unsubscribe();
    }
  }

  getModeDisplayName(): string {
    return 'AI Local';
  }

  // Etiqueta humana para la fuente del comando actual: "Normal" significa que el
  // controller (MPC) está mandando, "Recovery" cuando la rutina de curvas toma
  // control, "Señal" cuando el override del ManeuverManager (ex-signActions) está
  // activo a través de un escenario del BehaviorPlanner.
  getCommandSourceLabel(): string {
    const source = this.debugStatus?.command_source;
    const labels: { [key: string]: string } = {
      normal: 'Normal',
      curve_recovery: 'Recovery',
      sign_override: 'Señal',
      no_command: 'Sin comando',
      blocked_klem: 'Bloq. KL',
      blocked_inactive: 'Inactivo',
    };
    return source ? (labels[source] || source) : '--';
  }

  // Resumen del estado del actuador para que el operador entienda en una sola
  // línea por qué el robot no se mueve si está parado.
  getActuatorLabel(): string {
    if (this.debugStatus?.serial_connected === false) {
      return 'Serial desconectado';
    }
    if (this.debugStatus?.engine_enabled === false) {
      return 'Motor bloqueado';
    }
    if (this.debugStatus?.actuator_blocked_reason) {
      return this.debugStatus.actuator_blocked_reason;
    }
    return 'Habilitado';
  }

  onLocalAiSettingsChange(): void {
    this.localAiMinConfidence = Number(this.localAiMinConfidence);
    this.localAiInputSize = Number(this.localAiInputSize);
    this.localAiInterval = Number(this.localAiInterval);
    this.debouncedSendConfig();
  }

  setDebugView(viewId: number): void {
    this.selectedDebugView = viewId;
    if (viewId === 0) {
      this.debugImageSrc = null;
    }
    this.debouncedSendConfig();
  }

  getSelectedViewDescription(): string {
    const view = this.debugViews.find(v => v.id === this.selectedDebugView);
    return view ? view.description : '';
  }

  setStreamFps(fps: number): void {
    this.streamFps = fps;
    this.debouncedSendConfig();
  }

  setStreamQuality(quality: number): void {
    this.streamQuality = quality;
    this.debouncedSendConfig();
  }

  setStreamScale(scale: number): void {
    this.streamScale = scale;
    this.debouncedSendConfig();
  }

  toggleCurveRecovery(): void {
    this.useCurveRecovery = !this.useCurveRecovery;
    this.debouncedSendConfig();
  }

  toggleSection(section: string): void {
    this.expandedSections[section] = !this.expandedSections[section];
  }

  getSlidersByGroup(group: string): SliderConfig[] {
    return this.sliders.filter(s => s.group === group);
  }

  onSliderChange(slider: SliderConfig): void {
    // <input type="range"> entrega strings — forzar a Number antes de enviar al backend.
    slider.value = Number(slider.value);
    this.debouncedSendConfig();
  }

  // Debounce de 100 ms: agrupa cambios rápidos del operador (ej. arrastrar un slider)
  // en un único mensaje al backend, en vez de inundar el WebSocket.
  private debouncedSendConfig(): void {
    if (this.updateTimeout) {
      clearTimeout(this.updateTimeout);
    }
    this.updateTimeout = setTimeout(() => {
      this.sendConfig();
    }, 100);
  }

  sendConfig(): void {
    const config: { [key: string]: number | string } = {};

    config['detection_mode'] = this.selectedMode;
    config['local_ai_min_confidence'] = Number(this.localAiMinConfidence);
    config['local_ai_imgsz'] = Number(this.localAiInputSize);
    config['local_ai_interval'] = Number(this.localAiInterval);

    config['stream_debug_view'] = this.selectedDebugView;
    config['stream_debug_fps'] = this.streamFps;
    config['stream_debug_quality'] = this.streamQuality;
    config['stream_debug_scale'] = this.streamScale;

    config['use_curve_recovery'] = this.useCurveRecovery ? 1 : 0;

    for (const slider of this.sliders) {
      config[slider.key] = Number(slider.value);
    }

    localStorage.setItem('lineFollowingConfig', JSON.stringify(config));

    const message = JSON.stringify({
      Name: 'LineFollowingConfig',
      Value: config
    });

    this.webSocketService.sendMessageToFlask(message);
  }

  applyConfig(config: { [key: string]: any }): void {
    if (config['local_ai_min_confidence'] !== undefined) {
      this.localAiMinConfidence = Number(config['local_ai_min_confidence']);
    }
    if (config['local_ai_imgsz'] !== undefined) {
      this.localAiInputSize = Number(config['local_ai_imgsz']);
    }
    if (config['local_ai_interval'] !== undefined) {
      this.localAiInterval = Number(config['local_ai_interval']);
    }

    if (config['stream_debug_view'] !== undefined) {
      this.selectedDebugView = config['stream_debug_view'];
    }
    if (config['stream_debug_fps'] !== undefined) {
      this.streamFps = config['stream_debug_fps'];
    }
    if (config['stream_debug_quality'] !== undefined) {
      this.streamQuality = config['stream_debug_quality'];
    }
    if (config['stream_debug_scale'] !== undefined) {
      this.streamScale = config['stream_debug_scale'];
    }

    if (config['use_curve_recovery'] !== undefined) {
      this.useCurveRecovery = config['use_curve_recovery'] === 1;
    }

    for (const slider of this.sliders) {
      if (config[slider.key] !== undefined) {
        slider.value = Number(config[slider.key]);
      }
    }
  }

  resetDefaults(): void {
    this.localAiMinConfidence = 0.35;
    this.localAiInputSize = 320;
    this.localAiInterval = 0.04;

    this.selectedDebugView = 0;
    this.streamFps = 5;
    this.streamQuality = 50;
    this.streamScale = 0.5;

    this.useCurveRecovery = true;

    // Defaults sincronizados con threadLineFollowing.py — mover a backend cuando
    // BehaviorPlanner sea la fuente de verdad de speed (Fase 4).
    const defaults: { [key: string]: number } = {
      base_speed: 15,
      max_speed: 25,
      min_speed: 8,
      recovery_max_steer_frames: 6,
      recovery_reverse_speed: -8,
      recovery_reverse_time_min: 0.3,
      recovery_reverse_time_max: 1.5,
      recovery_reverse_steer_scale: 1.5,
      recovery_pre_turn_time: 0.6,
      recovery_realign_time: 0.6,
      recovery_error_shrink_ratio: 0.85,
    };

    for (const slider of this.sliders) {
      if (defaults[slider.key] !== undefined) {
        slider.value = defaults[slider.key];
      }
    }

    this.debugImageSrc = null;

    localStorage.removeItem('lineFollowingConfig');
    this.sendConfig();
  }
}
