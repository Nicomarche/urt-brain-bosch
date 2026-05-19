import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { ClusterService } from '../cluster.service';
import { WebSocketService } from '../../webSocket/web-socket.service';

interface StartupMoveStatus {
  state: 'empty' | 'ready' | 'recording' | 'replaying' | string;
  duration_s: number;
  samples: number;
  progress_s: number;
  progress_pct: number;
  can_record: boolean;
  error?: string | null;
}

@Component({
  selector: 'app-startup-move',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './startup-move.component.html',
  styleUrl: './startup-move.component.css'
})
export class StartupMoveComponent implements OnInit, OnDestroy {
  status: StartupMoveStatus = {
    state: 'empty',
    duration_s: 0,
    samples: 0,
    progress_s: 0,
    progress_pct: 0,
    can_record: false,
    error: null,
  };
  drivingMode: string = '';

  private statusSubscription: Subscription | undefined;
  private drivingModeSubscription: Subscription | undefined;

  constructor(
    private webSocketService: WebSocketService,
    private clusterService: ClusterService
  ) {}

  ngOnInit() {
    this.statusSubscription = this.webSocketService.receiveStartupMoveStatus().subscribe(
      (message) => {
        this.status = { ...this.status, ...(message.value || {}) };
      }
    );

    this.drivingModeSubscription = this.clusterService.drivingMode$.subscribe(
      (mode) => {
        this.drivingMode = mode;
      }
    );
  }

  ngOnDestroy() {
    this.statusSubscription?.unsubscribe();
    this.drivingModeSubscription?.unsubscribe();
  }

  get isRecording(): boolean {
    return this.status.state === 'recording';
  }

  get isReplaying(): boolean {
    return this.status.state === 'replaying';
  }

  get canStart(): boolean {
    return (this.drivingMode === 'manual' || this.status.can_record) && !this.isRecording && !this.isReplaying;
  }

  get canStop(): boolean {
    return this.isRecording;
  }

  get canClear(): boolean {
    return !this.isRecording && !this.isReplaying && this.status.state !== 'empty';
  }

  get primaryLabel(): string {
    return this.isRecording ? 'Detener' : 'Grabar inicio';
  }

  get statusLabel(): string {
    if (this.status.error) {
      return 'error';
    }
    if (this.isRecording) {
      return 'grabando';
    }
    if (this.isReplaying) {
      return 'reproduciendo';
    }
    if (this.status.state === 'ready') {
      return 'listo';
    }
    return 'vacio';
  }

  get durationLabel(): string {
    const duration = Number(this.status.duration_s || 0);
    return `${duration.toFixed(1)}s`;
  }

  get progressStyle(): string {
    const pct = Math.max(0, Math.min(100, Number(this.status.progress_pct || 0)));
    return `${pct}%`;
  }

  toggleRecording(): void {
    if (this.isRecording) {
      this.sendAction('stop');
      return;
    }
    if (this.canStart) {
      this.sendAction('start');
    }
  }

  clear(): void {
    if (this.canClear) {
      this.sendAction('clear');
    }
  }

  private sendAction(action: 'start' | 'stop' | 'clear'): void {
    this.webSocketService.sendMessageToFlask(
      JSON.stringify({ Name: 'StartupMoveControl', Value: { action } })
    );
  }
}
