import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { WebSocketService } from '../../webSocket/web-socket.service';

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
  ];

  private updateTimeout: any = null;

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
  }

  ngOnDestroy(): void {
    if (this.updateTimeout) {
      clearTimeout(this.updateTimeout);
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
    const config: { [key: string]: number } = {};
    
    for (const group of this.sliderGroups) {
      for (const slider of group.sliders) {
        config[slider.key] = slider.value;
      }
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
      base_speed: 10, max_speed: 10, min_speed: 5,
      kp: 1.5, kd: 0.3, smoothing_factor: 0.5, steering_sensitivity: 1.0,
      roi_height_start: 0.65, roi_height_end: 0.92, roi_width_margin_top: 0.35, roi_width_margin_bottom: 0.15,
      white_h_min: 81, white_h_max: 180, white_s_min: 0, white_s_max: 98, white_v_min: 200, white_v_max: 255,
      yellow_h_min: 173, yellow_h_max: 86, yellow_s_min: 100, yellow_s_max: 255, yellow_v_min: 100, yellow_v_max: 255,
      brightness: 5, contrast: 0.8, blur_kernel: 5, morph_kernel: 3,
      canny_low: 100, canny_high: 200, hough_threshold: 20, hough_min_line_length: 15, hough_max_line_gap: 200
    };
    
    this.applyConfig(defaults);
    localStorage.removeItem('lineFollowingConfig');
    this.sendConfig();
  }
}
