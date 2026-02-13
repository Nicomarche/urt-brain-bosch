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

interface LstrModel {
  id: number;
  name: string;
  resolution: string;
  speed: string;
}

interface DebugStatus {
  steering: number | null;
  speed: number | null;
  mode: string;
  view: string;
  active: boolean;
  lstr_available: boolean;
  hybridnets_connected?: boolean;
  hybridnets_roundtrip_ms?: number;
  hybridnets_server_fps?: number;
  supercombo_connected?: boolean;
  supercombo_roundtrip_ms?: number;
  supercombo_server_fps?: number;
}

@Component({
  selector: 'app-line-following',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './line-following.component.html',
  styleUrls: ['./line-following.component.css']
})
export class LineFollowingComponent implements OnInit, OnDestroy {
  
  // Mode selection
  selectedMode: string = 'opencv';
  lstrAvailable: boolean = true;  // Assume available by default, will be updated by backend
  
  // LSTR Model selection
  selectedLstrModel: number = 0;
  lstrModels: LstrModel[] = [
    { id: 0, name: 'Ultra Rápido', resolution: '180×320', speed: '~15 FPS' },
    { id: 1, name: 'Rápido', resolution: '240×320', speed: '~12 FPS' },
    { id: 2, name: 'Balanceado', resolution: '360×640', speed: '~8 FPS' },
    { id: 3, name: 'Preciso', resolution: '480×640', speed: '~5 FPS' },
    { id: 4, name: 'Máxima Calidad', resolution: '720×1280', speed: '~2 FPS' },
  ];

  // HybridNets AI Server settings
  hybridnetsServerUrl: string = 'ws://192.168.1.35:8500/ws/steering';
  hybridnetsJpegQuality: number = 70;
  hybridnetsTimeout: number = 2.0;
  hybridnetsConnected: boolean = false;
  hybridnetsRoundtripMs: number = 0;
  hybridnetsServerFps: number = 0;

  // Supercombo AI Server settings (openpilot model)
  supercomboServerUrl: string = 'ws://192.168.1.35:8500/ws/steering';
  supercomboJpegQuality: number = 70;
  supercomboTimeout: number = 2.0;
  supercomboConnected: boolean = false;
  supercomboRoundtripMs: number = 0;
  supercomboServerFps: number = 0;
  
  // Debug view selection
  selectedDebugView: number = 0;
  debugViews: DebugView[] = [
    { id: 0, name: 'Apagado', icon: '🚫', description: 'Stream de debug desactivado para máximo rendimiento.' },
    { id: 1, name: 'Final', icon: '🎯', description: 'Resultado final con líneas detectadas y trayectoria calculada.' },
    { id: 2, name: 'CLAHE', icon: '🌓', description: 'Imagen normalizada con CLAHE para mejor contraste.' },
    { id: 3, name: 'Ajustada', icon: '☀️', description: 'Imagen con brillo y contraste ajustados.' },
    { id: 4, name: 'HSV', icon: '🎨', description: 'Imagen convertida al espacio de color HSV.' },
    { id: 5, name: 'Blanco', icon: '⚪', description: 'Máscara de detección de líneas blancas.' },
    { id: 6, name: 'Amarillo', icon: '🟡', description: 'Máscara de detección de líneas amarillas.' },
    { id: 7, name: 'Combinada', icon: '🔀', description: 'Máscara combinada de blanco y amarillo.' },
    { id: 8, name: 'Bird Eye', icon: '🦅', description: 'Vista de pájaro (perspectiva superior).' },
    { id: 9, name: 'Ventana', icon: '📊', description: 'Ventana deslizante para detección de carriles.' },
    { id: 10, name: 'LSTR IA', icon: '🤖', description: 'Salida del modelo de IA LSTR con carriles detectados.' },
  ];
  
  // Stream settings
  streamFps: number = 5;
  streamQuality: number = 50;
  streamScale: number = 0.5;
  
  // Adaptive lighting toggles
  useClahe: boolean = true;
  useAdaptiveWhite: boolean = true;
  useGradientFallback: boolean = true;
  
  // Expandable sections
  expandedSections: { [key: string]: boolean } = {
    speed: false,
    pid: false,
    feedforward: false,
    roi: false,
    white: false,
    yellow: false,
    image: false,
    edge: false,
    adaptive: false
  };

  // All sliders organized by group
  sliders: SliderConfig[] = [
    // Speed
    { key: 'base_speed', label: 'Velocidad Base', min: 5, max: 40, step: 1, value: 10, group: 'speed' },
    { key: 'max_speed', label: 'Velocidad Máxima', min: 10, max: 50, step: 1, value: 10, group: 'speed' },
    { key: 'min_speed', label: 'Velocidad Mínima', min: 5, max: 30, step: 1, value: 5, group: 'speed' },
    // PID (basado en bfmc24-brain - ricardolopezb/bfmc24-brain)
    { key: 'max_error_px', label: 'Offset Máx (px→giro máx)', min: 10, max: 100, step: 1, value: 40, group: 'pid' },
    { key: 'kp', label: 'Proporcional (Kp)', min: 0, max: 50, step: 0.5, value: 25, group: 'pid' },
    { key: 'ki', label: 'Integral (Ki)', min: 0, max: 10, step: 0.1, value: 1.0, group: 'pid' },
    { key: 'kd', label: 'Derivativo (Kd)', min: 0, max: 20, step: 0.5, value: 4, group: 'pid' },
    { key: 'smoothing_factor', label: 'Suavizado', min: 0.1, max: 1.0, step: 0.05, value: 0.5, group: 'pid' },
    { key: 'max_steering', label: 'Ángulo Máx Giro', min: 5, max: 25, step: 1, value: 25, group: 'pid' },
    { key: 'lookahead', label: 'Lookahead (Anticipación)', min: 0.1, max: 0.8, step: 0.05, value: 0.4, group: 'pid' },
    { key: 'dead_zone_ratio', label: 'Zona Muerta', min: 0.0, max: 0.1, step: 0.005, value: 0.02, group: 'pid' },
    { key: 'integral_reset_interval', label: 'Reset Integral (cada N frames)', min: 1, max: 50, step: 1, value: 10, group: 'pid' },
    // Feed-Forward curve prediction
    { key: 'wheelbase', label: 'Distancia entre ejes (m)', min: 0.15, max: 0.35, step: 0.005, value: 0.265, group: 'feedforward' },
    { key: 'ff_weight', label: 'Peso Feed-Forward', min: 0.0, max: 1.0, step: 0.05, value: 0.6, group: 'feedforward' },
    { key: 'curvature_threshold', label: 'Umbral Curvatura', min: 0.1, max: 2.0, step: 0.1, value: 0.5, group: 'feedforward' },
    // ROI
    { key: 'roi_height_start', label: 'Inicio Altura', min: 0.3, max: 0.8, step: 0.05, value: 0.65, group: 'roi' },
    { key: 'roi_height_end', label: 'Fin Altura', min: 0.7, max: 1.0, step: 0.02, value: 0.92, group: 'roi' },
    { key: 'roi_width_margin_top', label: 'Margen Superior', min: 0.1, max: 0.5, step: 0.05, value: 0.35, group: 'roi' },
    { key: 'roi_width_margin_bottom', label: 'Margen Inferior', min: 0.0, max: 0.3, step: 0.05, value: 0.15, group: 'roi' },
    // White HSV
    { key: 'white_h_min', label: 'H Mín', min: 0, max: 180, step: 1, value: 81, group: 'white' },
    { key: 'white_h_max', label: 'H Máx', min: 0, max: 180, step: 1, value: 180, group: 'white' },
    { key: 'white_s_min', label: 'S Mín', min: 0, max: 255, step: 1, value: 0, group: 'white' },
    { key: 'white_s_max', label: 'S Máx', min: 0, max: 255, step: 1, value: 98, group: 'white' },
    { key: 'white_v_min', label: 'V Mín', min: 0, max: 255, step: 1, value: 200, group: 'white' },
    { key: 'white_v_max', label: 'V Máx', min: 0, max: 255, step: 1, value: 255, group: 'white' },
    // Yellow HSV
    { key: 'yellow_h_min', label: 'H Mín', min: 0, max: 180, step: 1, value: 173, group: 'yellow' },
    { key: 'yellow_h_max', label: 'H Máx', min: 0, max: 180, step: 1, value: 86, group: 'yellow' },
    { key: 'yellow_s_min', label: 'S Mín', min: 0, max: 255, step: 1, value: 100, group: 'yellow' },
    { key: 'yellow_s_max', label: 'S Máx', min: 0, max: 255, step: 1, value: 255, group: 'yellow' },
    { key: 'yellow_v_min', label: 'V Mín', min: 0, max: 255, step: 1, value: 100, group: 'yellow' },
    { key: 'yellow_v_max', label: 'V Máx', min: 0, max: 255, step: 1, value: 255, group: 'yellow' },
    // Image processing
    { key: 'brightness', label: 'Brillo', min: -100, max: 100, step: 5, value: 5, group: 'image' },
    { key: 'contrast', label: 'Contraste', min: 0.5, max: 2.0, step: 0.1, value: 0.8, group: 'image' },
    { key: 'blur_kernel', label: 'Kernel Blur', min: 1, max: 15, step: 2, value: 5, group: 'image' },
    { key: 'morph_kernel', label: 'Kernel Morph', min: 1, max: 10, step: 1, value: 3, group: 'image' },
    // Edge detection
    { key: 'canny_low', label: 'Canny Bajo', min: 10, max: 200, step: 10, value: 100, group: 'edge' },
    { key: 'canny_high', label: 'Canny Alto', min: 50, max: 300, step: 10, value: 200, group: 'edge' },
    { key: 'hough_threshold', label: 'Umbral Hough', min: 5, max: 100, step: 5, value: 20, group: 'edge' },
    { key: 'hough_min_line_length', label: 'Long. Mín Línea', min: 5, max: 100, step: 5, value: 15, group: 'edge' },
    { key: 'hough_max_line_gap', label: 'Espacio Máx Línea', min: 10, max: 300, step: 10, value: 200, group: 'edge' },
    // Adaptive lighting
    { key: 'clahe_clip_limit', label: 'CLAHE Clip', min: 1.0, max: 5.0, step: 0.5, value: 2.0, group: 'adaptive' },
    { key: 'clahe_grid_size', label: 'CLAHE Grid', min: 4, max: 16, step: 2, value: 8, group: 'adaptive' },
    { key: 'adaptive_white_percentile', label: 'Percentil Blanco', min: 80, max: 98, step: 1, value: 92, group: 'adaptive' },
    { key: 'adaptive_white_min_threshold', label: 'Umbral Mín Blanco', min: 150, max: 220, step: 5, value: 180, group: 'adaptive' },
    { key: 'gradient_percentile', label: 'Percentil Gradiente', min: 75, max: 95, step: 1, value: 85, group: 'adaptive' },
  ];

  private updateTimeout: any = null;
  private debugStreamSubscription: Subscription | null = null;
  private debugStatusSubscription: Subscription | null = null;

  // Debug stream properties
  debugImageSrc: string | null = null;
  debugStatus: DebugStatus | null = null;
  debugStreamEnabled: boolean = true;

  get debugViewName(): string {
    const view = this.debugViews.find(v => v.id === this.selectedDebugView);
    return view ? view.name : 'Apagado';
  }

  constructor(private webSocketService: WebSocketService) {}

  ngOnInit(): void {
    // Load saved config from localStorage if available
    const savedConfig = localStorage.getItem('lineFollowingConfig');
    if (savedConfig) {
      try {
        const config = JSON.parse(savedConfig);
        this.applyConfig(config);
      } catch (e) {
        console.error('Failed to load saved config:', e);
      }
    }

    // Subscribe to debug stream
    this.debugStreamSubscription = this.webSocketService
      .receiveLineFollowingDebug()
      .subscribe((imageData: string) => {
        if (imageData) {
          this.debugImageSrc = 'data:image/jpeg;base64,' + imageData;
        }
      });

    // Subscribe to debug status
    this.debugStatusSubscription = this.webSocketService
      .receiveLineFollowingStatus()
      .subscribe((status: DebugStatus) => {
        this.debugStatus = status;
        this.lstrAvailable = status?.lstr_available ?? false;
        // Update HybridNets connection status
        if (status?.hybridnets_connected !== undefined) {
          this.hybridnetsConnected = status.hybridnets_connected;
        }
        if (status?.hybridnets_roundtrip_ms !== undefined) {
          this.hybridnetsRoundtripMs = status.hybridnets_roundtrip_ms;
        }
        if (status?.hybridnets_server_fps !== undefined) {
          this.hybridnetsServerFps = status.hybridnets_server_fps;
        }
        // Update Supercombo connection status
        if (status?.supercombo_connected !== undefined) {
          this.supercomboConnected = status.supercombo_connected;
        }
        if (status?.supercombo_roundtrip_ms !== undefined) {
          this.supercomboRoundtripMs = status.supercombo_roundtrip_ms;
        }
        if (status?.supercombo_server_fps !== undefined) {
          this.supercomboServerFps = status.supercombo_server_fps;
        }
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

  // Mode methods
  setMode(mode: string): void {
    // HybridNets is always available (remote server), LSTR/hybrid need local LSTR
    if ((mode === 'lstr' || mode === 'hybrid') && !this.lstrAvailable) return;
    this.selectedMode = mode;
    this.debouncedSendConfig();
  }

  getModeDisplayName(): string {
    const names: { [key: string]: string } = {
      'opencv': 'OpenCV',
      'lstr': 'LSTR IA',
      'hybrid': 'Híbrido',
      'hybridnets': 'HybridNets',
      'supercombo': 'Supercombo'
    };
    return names[this.selectedMode] || this.selectedMode;
  }

  // HybridNets methods
  setHybridnetsServerUrl(url: string): void {
    this.hybridnetsServerUrl = url;
    this.debouncedSendConfig();
  }

  setHybridnetsJpegQuality(quality: number): void {
    this.hybridnetsJpegQuality = quality;
    this.debouncedSendConfig();
  }

  setHybridnetsTimeout(timeout: number): void {
    this.hybridnetsTimeout = timeout;
    this.debouncedSendConfig();
  }

  setHybridnetsEndpoint(endpoint: string): void {
    const base = this.hybridnetsServerUrl.replace(/\/ws\/.*$/, '');
    this.hybridnetsServerUrl = base + endpoint;
    this.debouncedSendConfig();
  }

  // Supercombo methods
  setSupercomboServerUrl(url: string): void {
    this.supercomboServerUrl = url;
    this.debouncedSendConfig();
  }

  setSupercomboJpegQuality(quality: number): void {
    this.supercomboJpegQuality = quality;
    this.debouncedSendConfig();
  }

  setSupercomboTimeout(timeout: number): void {
    this.supercomboTimeout = timeout;
    this.debouncedSendConfig();
  }

  setSupercomboEndpoint(endpoint: string): void {
    const base = this.supercomboServerUrl.replace(/\/ws\/.*$/, '');
    this.supercomboServerUrl = base + endpoint;
    this.debouncedSendConfig();
  }

  // LSTR Model methods
  setLstrModel(modelId: number): void {
    this.selectedLstrModel = modelId;
    this.debouncedSendConfig();
  }

  // Debug view methods
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

  // Stream settings methods
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

  // Toggle methods
  toggleClahe(): void {
    this.useClahe = !this.useClahe;
    this.debouncedSendConfig();
  }

  toggleAdaptiveWhite(): void {
    this.useAdaptiveWhite = !this.useAdaptiveWhite;
    this.debouncedSendConfig();
  }

  toggleGradientFallback(): void {
    this.useGradientFallback = !this.useGradientFallback;
    this.debouncedSendConfig();
  }

  // Section toggle
  toggleSection(section: string): void {
    this.expandedSections[section] = !this.expandedSections[section];
  }

  // Get sliders by group
  getSlidersByGroup(group: string): SliderConfig[] {
    return this.sliders.filter(s => s.group === group);
  }

  onSliderChange(slider: SliderConfig): void {
    this.debouncedSendConfig();
  }

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
    
    // Add mode settings
    config['detection_mode'] = this.selectedMode;
    config['lstr_model_size'] = this.selectedLstrModel;
    
    // Add HybridNets settings
    config['hybridnets_server_url'] = this.hybridnetsServerUrl;
    config['hybridnets_jpeg_quality'] = this.hybridnetsJpegQuality;
    config['hybridnets_timeout'] = this.hybridnetsTimeout;
    
    // Add Supercombo settings
    config['supercombo_server_url'] = this.supercomboServerUrl;
    config['supercombo_jpeg_quality'] = this.supercomboJpegQuality;
    config['supercombo_timeout'] = this.supercomboTimeout;
    
    // Add stream settings
    config['stream_debug_view'] = this.selectedDebugView;
    config['stream_debug_fps'] = this.streamFps;
    config['stream_debug_quality'] = this.streamQuality;
    config['stream_debug_scale'] = this.streamScale;
    
    // Add toggle settings
    config['use_clahe'] = this.useClahe ? 1 : 0;
    config['use_adaptive_white'] = this.useAdaptiveWhite ? 1 : 0;
    config['use_gradient_fallback'] = this.useGradientFallback ? 1 : 0;
    
    // Add all slider values
    for (const slider of this.sliders) {
      config[slider.key] = slider.value;
    }
    
    // Save to localStorage
    localStorage.setItem('lineFollowingConfig', JSON.stringify(config));
    
    // Send to backend
    const message = JSON.stringify({
      Name: 'LineFollowingConfig',
      Value: config
    });
    
    this.webSocketService.sendMessageToFlask(message);
  }

  applyConfig(config: { [key: string]: any }): void {
    // Apply mode
    if (config['detection_mode']) {
      this.selectedMode = config['detection_mode'];
    }
    if (config['lstr_model_size'] !== undefined) {
      this.selectedLstrModel = config['lstr_model_size'];
    }
    
    // Apply HybridNets settings
    if (config['hybridnets_server_url']) {
      this.hybridnetsServerUrl = config['hybridnets_server_url'];
    }
    if (config['hybridnets_jpeg_quality'] !== undefined) {
      this.hybridnetsJpegQuality = config['hybridnets_jpeg_quality'];
    }
    if (config['hybridnets_timeout'] !== undefined) {
      this.hybridnetsTimeout = config['hybridnets_timeout'];
    }
    
    // Apply Supercombo settings
    if (config['supercombo_server_url']) {
      this.supercomboServerUrl = config['supercombo_server_url'];
    }
    if (config['supercombo_jpeg_quality'] !== undefined) {
      this.supercomboJpegQuality = config['supercombo_jpeg_quality'];
    }
    if (config['supercombo_timeout'] !== undefined) {
      this.supercomboTimeout = config['supercombo_timeout'];
    }
    
    // Apply stream settings
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
    
    // Apply toggles
    if (config['use_clahe'] !== undefined) {
      this.useClahe = config['use_clahe'] === 1;
    }
    if (config['use_adaptive_white'] !== undefined) {
      this.useAdaptiveWhite = config['use_adaptive_white'] === 1;
    }
    if (config['use_gradient_fallback'] !== undefined) {
      this.useGradientFallback = config['use_gradient_fallback'] === 1;
    }
    
    // Apply sliders
    for (const slider of this.sliders) {
      if (config[slider.key] !== undefined) {
        slider.value = config[slider.key];
      }
    }
  }

  resetDefaults(): void {
    // Reset mode
    this.selectedMode = 'opencv';
    this.selectedLstrModel = 0;
    
    // Reset HybridNets
    this.hybridnetsServerUrl = 'ws://192.168.1.35:8500/ws/steering';
    this.hybridnetsJpegQuality = 70;
    this.hybridnetsTimeout = 2.0;
    
    // Reset Supercombo
    this.supercomboServerUrl = 'ws://192.168.1.35:8500/ws/steering';
    this.supercomboJpegQuality = 70;
    this.supercomboTimeout = 2.0;
    
    // Reset stream
    this.selectedDebugView = 0;
    this.streamFps = 5;
    this.streamQuality = 50;
    this.streamScale = 0.5;
    
    // Reset toggles
    this.useClahe = true;
    this.useAdaptiveWhite = true;
    this.useGradientFallback = true;
    
    // Reset sliders to defaults
    const defaults: { [key: string]: number } = {
      base_speed: 10, max_speed: 10, min_speed: 5,
      max_error_px: 40, kp: 25, ki: 1.0, kd: 4, smoothing_factor: 0.5,
      max_steering: 25, lookahead: 0.4, dead_zone_ratio: 0.02, integral_reset_interval: 10,
      wheelbase: 0.265, ff_weight: 0.6, curvature_threshold: 0.5,
      roi_height_start: 0.65, roi_height_end: 0.92, roi_width_margin_top: 0.35, roi_width_margin_bottom: 0.15,
      white_h_min: 81, white_h_max: 180, white_s_min: 0, white_s_max: 98, white_v_min: 200, white_v_max: 255,
      yellow_h_min: 173, yellow_h_max: 86, yellow_s_min: 100, yellow_s_max: 255, yellow_v_min: 100, yellow_v_max: 255,
      brightness: 5, contrast: 0.8, blur_kernel: 5, morph_kernel: 3,
      canny_low: 100, canny_high: 200, hough_threshold: 20, hough_min_line_length: 15, hough_max_line_gap: 200,
      clahe_clip_limit: 2.0, clahe_grid_size: 8,
      adaptive_white_percentile: 92, adaptive_white_min_threshold: 180,
      gradient_percentile: 85
    };
    
    for (const slider of this.sliders) {
      if (defaults[slider.key] !== undefined) {
        slider.value = defaults[slider.key];
      }
    }
    
    // Clear image
    this.debugImageSrc = null;
    
    localStorage.removeItem('lineFollowingConfig');
    this.sendConfig();
  }
}
