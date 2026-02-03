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
}

interface SliderGroup {
  title: string;
  sliders: SliderConfig[];
}

interface DebugStatus {
  steering: number | null;
  speed: number | null;
  mode: string;
  view: string;
  active: boolean;
  lstr_available: boolean;
}

@Component({
  selector: 'app-line-following',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './line-following.component.html',
  styleUrls: ['./line-following.component.css']
})
export class LineFollowingComponent implements OnInit, OnDestroy {
  
  sliderGroups: SliderGroup[] = [
    {
      title: '🤖 Detection Mode',
      sliders: [
        { key: 'detection_mode', label: 'Mode (0=OpenCV, 1=LSTR AI, 2=Hybrid)', min: 0, max: 2, step: 1, value: 0 },
        { key: 'lstr_model_size', label: 'LSTR Model (0=180x320 Fast → 4=720x1280 Accurate)', min: 0, max: 4, step: 1, value: 0 },
      ]
    },
    {
      title: '📺 Debug Stream (Web UI)',
      sliders: [
        { key: 'stream_debug_view', label: 'View (0=Off, 1=Final, 2=CLAHE, 3=Adjusted, 4=HSV, 5=White, 6=Yellow, 7=Combined, 8=BirdEye, 9=SlidingWin, 10=LSTR)', min: 0, max: 10, step: 1, value: 0 },
        { key: 'stream_debug_fps', label: 'Stream FPS', min: 1, max: 15, step: 1, value: 5 },
        { key: 'stream_debug_quality', label: 'JPEG Quality', min: 20, max: 90, step: 10, value: 50 },
        { key: 'stream_debug_scale', label: 'Scale (0.25=25%, 0.5=50%, 1=100%)', min: 0.25, max: 1.0, step: 0.25, value: 0.5 },
      ]
    },
    {
      title: '🚗 Speed',
      sliders: [
        { key: 'base_speed', label: 'Base Speed', min: 5, max: 40, step: 1, value: 10 },
        { key: 'max_speed', label: 'Max Speed', min: 10, max: 50, step: 1, value: 10 },
        { key: 'min_speed', label: 'Min Speed', min: 5, max: 30, step: 1, value: 5 },
      ]
    },
    {
      title: '🎯 PID Control',
      sliders: [
        { key: 'kp', label: 'Proportional (Kp)', min: 0.1, max: 5.0, step: 0.1, value: 1.5 },
        { key: 'kd', label: 'Derivative (Kd)', min: 0.0, max: 2.0, step: 0.1, value: 0.3 },
        { key: 'smoothing_factor', label: 'Smoothing', min: 0.1, max: 1.0, step: 0.1, value: 0.5 },
        { key: 'steering_sensitivity', label: 'Sensitivity', min: 0.5, max: 2.0, step: 0.1, value: 1.0 },
      ]
    },
    {
      title: '📐 ROI (Region of Interest)',
      sliders: [
        { key: 'roi_height_start', label: 'Height Start', min: 0.3, max: 0.8, step: 0.05, value: 0.65 },
        { key: 'roi_height_end', label: 'Height End', min: 0.7, max: 1.0, step: 0.02, value: 0.92 },
        { key: 'roi_width_margin_top', label: 'Top Margin', min: 0.1, max: 0.5, step: 0.05, value: 0.35 },
        { key: 'roi_width_margin_bottom', label: 'Bottom Margin', min: 0.0, max: 0.3, step: 0.05, value: 0.15 },
      ]
    },
    {
      title: '⚪ White Line HSV',
      sliders: [
        { key: 'white_h_min', label: 'H Min', min: 0, max: 180, step: 1, value: 81 },
        { key: 'white_h_max', label: 'H Max', min: 0, max: 180, step: 1, value: 180 },
        { key: 'white_s_min', label: 'S Min', min: 0, max: 255, step: 1, value: 0 },
        { key: 'white_s_max', label: 'S Max', min: 0, max: 255, step: 1, value: 98 },
        { key: 'white_v_min', label: 'V Min', min: 0, max: 255, step: 1, value: 200 },
        { key: 'white_v_max', label: 'V Max', min: 0, max: 255, step: 1, value: 255 },
      ]
    },
    {
      title: '🟡 Yellow Line HSV',
      sliders: [
        { key: 'yellow_h_min', label: 'H Min', min: 0, max: 180, step: 1, value: 173 },
        { key: 'yellow_h_max', label: 'H Max', min: 0, max: 180, step: 1, value: 86 },
        { key: 'yellow_s_min', label: 'S Min', min: 0, max: 255, step: 1, value: 100 },
        { key: 'yellow_s_max', label: 'S Max', min: 0, max: 255, step: 1, value: 255 },
        { key: 'yellow_v_min', label: 'V Min', min: 0, max: 255, step: 1, value: 100 },
        { key: 'yellow_v_max', label: 'V Max', min: 0, max: 255, step: 1, value: 255 },
      ]
    },
    {
      title: '🔧 Image Processing',
      sliders: [
        { key: 'brightness', label: 'Brightness', min: -100, max: 100, step: 5, value: 5 },
        { key: 'contrast', label: 'Contrast', min: 0.5, max: 2.0, step: 0.1, value: 0.8 },
        { key: 'blur_kernel', label: 'Blur Kernel', min: 1, max: 15, step: 2, value: 5 },
        { key: 'morph_kernel', label: 'Morph Kernel', min: 1, max: 10, step: 1, value: 3 },
      ]
    },
    {
      title: '📏 Edge Detection',
      sliders: [
        { key: 'canny_low', label: 'Canny Low', min: 10, max: 200, step: 10, value: 100 },
        { key: 'canny_high', label: 'Canny High', min: 50, max: 300, step: 10, value: 200 },
        { key: 'hough_threshold', label: 'Hough Threshold', min: 5, max: 100, step: 5, value: 20 },
        { key: 'hough_min_line_length', label: 'Min Line Length', min: 5, max: 100, step: 5, value: 15 },
        { key: 'hough_max_line_gap', label: 'Max Line Gap', min: 10, max: 300, step: 10, value: 200 },
      ]
    },
    {
      title: '💡 Adaptive Lighting (NEW)',
      sliders: [
        { key: 'use_clahe', label: 'Enable CLAHE', min: 0, max: 1, step: 1, value: 1 },
        { key: 'clahe_clip_limit', label: 'CLAHE Clip Limit', min: 1.0, max: 5.0, step: 0.5, value: 2.0 },
        { key: 'clahe_grid_size', label: 'CLAHE Grid Size', min: 4, max: 16, step: 2, value: 8 },
        { key: 'use_adaptive_white', label: 'Adaptive White', min: 0, max: 1, step: 1, value: 1 },
        { key: 'adaptive_white_percentile', label: 'White Percentile', min: 80, max: 98, step: 1, value: 92 },
        { key: 'adaptive_white_min_threshold', label: 'Min White Threshold', min: 150, max: 220, step: 5, value: 180 },
        { key: 'use_gradient_fallback', label: 'Gradient Fallback', min: 0, max: 1, step: 1, value: 1 },
        { key: 'gradient_percentile', label: 'Gradient Percentile', min: 75, max: 95, step: 1, value: 85 },
      ]
    },
  ];

  private updateTimeout: any = null;
  private debugStreamSubscription: Subscription | null = null;
  private debugStatusSubscription: Subscription | null = null;

  // Debug stream properties
  debugImageSrc: string | null = null;
  debugStatus: DebugStatus | null = null;
  debugStreamEnabled: boolean = true;

  // View name mapping
  private viewNames: { [key: number]: string } = {
    0: 'Off',
    1: 'Final Result',
    2: 'CLAHE Normalized',
    3: 'Brightness/Contrast',
    4: 'HSV Color Space',
    5: 'White Mask',
    6: 'Yellow Mask',
    7: 'Combined Mask',
    8: "Bird's Eye View",
    9: 'Sliding Window',
    10: 'LSTR AI'
  };

  get debugViewName(): string {
    const viewSlider = this.sliderGroups
      .find(g => g.title.includes('Debug Stream'))
      ?.sliders.find(s => s.key === 'stream_debug_view');
    return viewSlider ? (this.viewNames[viewSlider.value] || 'Unknown') : 'Off';
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

  onSliderChange(slider: SliderConfig): void {
    // Debounce the updates to avoid flooding the websocket
    if (this.updateTimeout) {
      clearTimeout(this.updateTimeout);
    }
    
    this.updateTimeout = setTimeout(() => {
      this.sendConfig();
    }, 100);
  }

  sendConfig(): void {
    const config: { [key: string]: number | string } = {};
    
    for (const group of this.sliderGroups) {
      for (const slider of group.sliders) {
        config[slider.key] = slider.value;
      }
    }
    
    // Convert detection_mode number to string
    const modeMap: { [key: number]: string } = { 0: 'opencv', 1: 'lstr', 2: 'hybrid' };
    if (config['detection_mode'] !== undefined) {
      config['detection_mode'] = modeMap[config['detection_mode'] as number] || 'opencv';
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

  applyConfig(config: { [key: string]: number }): void {
    for (const group of this.sliderGroups) {
      for (const slider of group.sliders) {
        if (config[slider.key] !== undefined) {
          slider.value = config[slider.key];
        }
      }
    }
  }

  resetDefaults(): void {
    // Reset to default values
    const defaults: { [key: string]: number } = {
      // Detection mode (0=opencv, 1=lstr, 2=hybrid)
      detection_mode: 0,
      lstr_model_size: 0,  // 0=180x320 (fastest)
      // Debug streaming
      stream_debug_view: 0, stream_debug_fps: 5, stream_debug_quality: 50, stream_debug_scale: 0.5,
      base_speed: 10, max_speed: 10, min_speed: 5,
      kp: 1.5, kd: 0.3, smoothing_factor: 0.5, steering_sensitivity: 1.0,
      roi_height_start: 0.65, roi_height_end: 0.92, roi_width_margin_top: 0.35, roi_width_margin_bottom: 0.15,
      white_h_min: 81, white_h_max: 180, white_s_min: 0, white_s_max: 98, white_v_min: 200, white_v_max: 255,
      yellow_h_min: 173, yellow_h_max: 86, yellow_s_min: 100, yellow_s_max: 255, yellow_v_min: 100, yellow_v_max: 255,
      brightness: 5, contrast: 0.8, blur_kernel: 5, morph_kernel: 3,
      canny_low: 100, canny_high: 200, hough_threshold: 20, hough_min_line_length: 15, hough_max_line_gap: 200,
      // Adaptive lighting defaults
      use_clahe: 1, clahe_clip_limit: 2.0, clahe_grid_size: 8,
      use_adaptive_white: 1, adaptive_white_percentile: 92, adaptive_white_min_threshold: 180,
      use_gradient_fallback: 1, gradient_percentile: 85
    };
    
    this.applyConfig(defaults);
    localStorage.removeItem('lineFollowingConfig');
    this.sendConfig();
  }
}
