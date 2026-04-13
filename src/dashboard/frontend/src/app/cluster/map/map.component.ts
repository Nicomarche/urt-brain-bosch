// Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

//  1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, Input, ViewChild } from '@angular/core';
import { Subscription } from 'rxjs';

import { WebSocketService } from '../../webSocket/web-socket.service';
import { ClusterService } from '../cluster.service';
import { MapCursorComponent } from './map-cursor/map-cursor.component';
import { MapSemaphoreComponent } from './map-semaphore/map-semaphore.component';

interface Semaphore {
  x: number;
  y: number;
  state: string;
}

interface RoutePoint {
  x: number;
  y: number;
}

interface MapBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

interface DestinationOption {
  id: string;
  label: string;
  node_id: string;
}

interface MapNode {
  id: string;
  node_id: string;
  x: number;
  y: number;
  yaw: number;
  attr: number;
  is_start: boolean;
  label: string;
  semantic_types: string[];
  percentX: number;
  percentY: number;
}

@Component({
  selector: 'app-map',
  standalone: true,
  imports: [MapCursorComponent, MapSemaphoreComponent, CommonModule],
  templateUrl: './map.component.html',
  styleUrl: './map.component.css'
})
export class MapComponent {
  public trackWidthMeters = 20.67;
  public trackHeightMeters = 13.76;
  public mapImageSrc = '/assets/TrackPreview.jpg';

  @Input() cursorRotation: number = 0;

  @ViewChild('imageElement') imageElementRef!: ElementRef<HTMLImageElement>;
  @ViewChild('imageContainer') imageContainerRef!: ElementRef<HTMLElement>;

  private mapX: number = 0;
  private mapY: number = 0;

  private screenSize = { width: 100, height: 100 };
  private mapSize: number = 500;
  private mapWidth: number = 0;
  private mapHeight: number = 0;

  private cursorSize: number = 6;
  private semaphoreSize: number = 3;

  private semaphoreXOffset: number = 10;
  private semaphoreYOffset: number = 1.45;
  private worldBounds: MapBounds = {
    xMin: 0,
    xMax: this.trackWidthMeters,
    yMin: 0,
    yMax: this.trackHeightMeters,
  };
  private yAxisInverted = true;

  public semaphores: Map<number, Semaphore> = new Map<number, Semaphore>();
  public routePoints: RoutePoint[] = [];
  public mapNodes: MapNode[] = [];
  public routeActive: boolean = false;
  public routePolylinePoints: string = '';
  public destinationPoint: RoutePoint | null = null;
  public destinationPercentX: number = 0;
  public destinationPercentY: number = 0;
  public navigationSummary: string = 'Mapa libre';
  public nextSemanticSummary: string = 'Sin evento próximo';
  public relocalizationSummary: string = 'map_match';
  public availableDestinations: DestinationOption[] = [];
  public selectedDestinationId: string = '';
  public routeQueueSummary: string = '';
  public currentDrivingMode: string = '';
  public currentNodeId: string = '';
  public upcomingNodeId: string = '';
  public lastManualGpsNodeId: string = '';

  private locationSubscription: Subscription | undefined;
  private semaphoresAndCarsSubscription: Subscription | undefined;
  private navigationSubscription: Subscription | undefined;
  private drivingModeSubscription: Subscription | undefined;

  constructor(
    private webSocketService: WebSocketService,
    private clusterService: ClusterService,
  ) { }

  ngOnInit() {
    this.locationSubscription = this.webSocketService.receiveLocation().subscribe(
      (message) => {
        const point = this.toMapPercent({
          x: Number(message?.value?.x ?? 0),
          y: Number(message?.value?.y ?? 0),
        });
        this.mapX = point.x;
        this.mapY = point.y;
        this.updateMap();
      },
    );

    this.semaphoresAndCarsSubscription = this.webSocketService.receiveSemaphores().subscribe(
      (message) => {
        const recv = message.value;
        this.semaphores.set(recv.id, { x: recv.x, y: recv.y, state: recv.state });
        this.updateMap();
      },
    );

    this.navigationSubscription = this.webSocketService.receiveNavigationStatus().subscribe(
      (message) => {
        const value = message?.value ?? message ?? {};
        this.routeActive = Boolean(value.route_active);
        this.applyMapMetadata(value.map_metadata);
        this.currentNodeId = String(value.current_node_id ?? '');
        this.upcomingNodeId = String(value.upcoming_node_id ?? '');

        this.routePoints = Array.isArray(value.route_points)
          ? value.route_points.map((point: any) => ({
              x: Number(point?.x ?? 0),
              y: Number(point?.y ?? 0),
            }))
          : [];

        this.routePolylinePoints = this.routePoints
          .map((point) => {
            const percent = this.toMapPercent(point);
            return `${percent.x},${percent.y}`;
          })
          .join(' ');

        const destination = value.destination_point;
        if (destination && destination.x !== undefined && destination.y !== undefined) {
          this.destinationPoint = {
            x: Number(destination.x),
            y: Number(destination.y),
          };
          const percent = this.toMapPercent(this.destinationPoint);
          this.destinationPercentX = percent.x;
          this.destinationPercentY = percent.y;
        } else {
          this.destinationPoint = null;
        }

        this.mapNodes = Array.isArray(value.map_nodes)
          ? value.map_nodes
              .map((node: any) => this.toMapNode(node))
              .filter((node: MapNode | null): node is MapNode => node !== null)
          : [];

        this.availableDestinations = Array.isArray(value.available_destinations)
          ? value.available_destinations.map((item: any) => ({
              id: String(item?.id ?? ''),
              label: String(item?.label ?? item?.id ?? ''),
              node_id: String(item?.node_id ?? ''),
            })).filter((item: DestinationOption) => Boolean(item.id && item.node_id))
          : [];
        if (!this.selectedDestinationId && this.availableDestinations.length > 0) {
          this.selectedDestinationId = this.availableDestinations[0].id;
        }

        const queue = Array.isArray(value.route_queue ?? value.queue)
          ? (value.route_queue ?? value.queue)
          : [];
        this.routeQueueSummary = queue
          .map((item: any) => String(item?.label ?? item?.id ?? item?.node_id ?? ''))
          .filter((label: string) => Boolean(label))
          .join(' → ');

        const destinationLabel = value.destination_label || value.destination_node_id || value.destination || 'loop';
        const maneuverLabel = value.maneuver_type || 'none';
        const nextSemanticLabel = value.next_semantic_label || value.next_semantic_type || 'sin evento';
        const relocalization = value.relocalization_mode || 'map_match';
        const relocalizationSource = value.last_relocalization_source ? ` · ${value.last_relocalization_source}` : '';
        const progressPct = Math.round(Number(value.route_progress ?? value.progress ?? 0) * 100);
        this.navigationSummary = this.routeActive
          ? `Ruta ${destinationLabel} · ${maneuverLabel} · ${progressPct}%`
          : 'Mapa libre';
        this.nextSemanticSummary = `Próximo: ${nextSemanticLabel}`;
        this.relocalizationSummary = `Reloc: ${relocalization}${relocalizationSource}`;

        this.updateMap();
      },
    );

    this.drivingModeSubscription = this.clusterService.drivingMode$.subscribe(
      (value) => {
        this.currentDrivingMode = String(value ?? '');
      },
    );

    this.updateMap();
  }

  ngOnDestroy() {
    if (this.locationSubscription) {
      this.locationSubscription.unsubscribe();
    }
    if (this.semaphoresAndCarsSubscription) {
      this.semaphoresAndCarsSubscription.unsubscribe();
    }
    if (this.navigationSubscription) {
      this.navigationSubscription.unsubscribe();
    }
    if (this.drivingModeSubscription) {
      this.drivingModeSubscription.unsubscribe();
    }
    this.webSocketService.disconnectSocket();
  }

  @HostListener('window:resize')
  onWindowResize(): void {
    this.updateMap();
  }

  onLoadTrack(image: HTMLImageElement): void {
    const imageContainer = document.getElementById('map-track-image-container') as HTMLElement;
    if (imageContainer) {
      imageContainer.style.width = `${this.screenSize.width}%`;
      imageContainer.style.height = `${this.screenSize.height}%`;
    }

    this.mapWidth = image.width;
    this.mapHeight = image.height;

    const map = document.getElementById('map-track-image') as HTMLElement;
    const overlay = document.getElementById('map-route-overlay') as HTMLElement;
    if (map) {
      map.style.width = `${this.mapSize}%`;
      map.style.height = 'auto';
      this.mapWidth = this.mapSize;
    }
    if (overlay) {
      overlay.style.width = `${this.mapSize}%`;
    }
  }

  onLoadCursor(): void {
    const cursor = document.getElementById('map-cursor') as HTMLElement;
    if (cursor) {
      cursor.style.width = `${this.cursorSize}%`;
      cursor.style.height = 'auto';
    }
  }

  onLoadSemaphore(id: number): void {
    const semaphore = document.getElementById('map-semaphore' + id) as HTMLElement;
    if (semaphore) {
      semaphore.style.position = 'absolute';
      semaphore.style.width = `${this.semaphoreSize}%`;
      semaphore.style.height = 'auto';
      this.updateMap();
    }
  }

  updateMap(): void {
    const map = document.getElementById('map-track-image') as HTMLElement;
    if (!map) {
      return;
    }

    let imageContainerHeight = 0;
    if (this.imageContainerRef) {
      const imgContainer = this.imageContainerRef.nativeElement;
      const rect = imgContainer.getBoundingClientRect();
      imageContainerHeight = rect.height;
    }

    if (this.imageElementRef) {
      const image = this.imageElementRef.nativeElement;
      this.mapWidth = this.mapSize;
      if (imageContainerHeight > 0) {
        this.mapHeight = (100 * image.height) / imageContainerHeight;
      }
    }

    const top = (this.mapY * this.mapHeight) / 100 - this.mapHeight - (this.screenSize.height / 2 - this.mapHeight);
    const left = (this.mapX * this.mapWidth) / 100 - this.mapWidth - (this.screenSize.width / 2 - this.mapWidth);

    map.style.top = `${-top}%`;
    map.style.left = `${-left}%`;

    const overlay = document.getElementById('map-route-overlay') as HTMLElement;
    if (overlay) {
      overlay.style.top = `${-top}%`;
      overlay.style.left = `${-left}%`;
      overlay.style.width = `${this.mapSize}%`;
      overlay.style.height = `${this.mapHeight}%`;
    }

    this.semaphores.forEach((value: Semaphore, key: number) => {
      const semaphore = document.getElementById('map-semaphore' + key) as HTMLElement;
      if (!semaphore) {
        return;
      }

      const x = (value.x * 100 / this.trackWidthMeters);
      const y = (value.y * 100 / this.trackHeightMeters);

      const topNew = (y * this.mapHeight) / 100;
      const leftNew = (x * this.mapWidth) / 100;

      semaphore.style.top = `${(-top - this.semaphoreXOffset) + topNew}%`;
      semaphore.style.left = `${(-left - this.semaphoreYOffset) + leftNew}%`;
    });
  }

  onMapClick(event: MouseEvent): void {
    if (this.isAutoMode) {
      return;
    }
    if (!this.imageElementRef) {
      return;
    }

    const map = this.imageElementRef.nativeElement;
    const rect = map.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }

    const xRatio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const yRatio = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    const localY = this.yAxisInverted
      ? (1 - yRatio) * this.trackHeightMeters
      : yRatio * this.trackHeightMeters;

    const destination = {
      x: Number((this.worldBounds.xMin + xRatio * this.trackWidthMeters).toFixed(4)),
      y: Number((this.worldBounds.yMin + localY).toFixed(4)),
    };

    this.webSocketService.sendMessageToFlask(JSON.stringify({
      Name: 'NavigationCommand',
      Value: {
        mode: 'go_to',
        destinations: [destination],
      },
    }));
  }

  onNodeClick(node: MapNode, event: MouseEvent): void {
    if (!this.isAutoMode) {
      return;
    }
    event.stopPropagation();
    const fix = this.buildLocalisationFix(node);
    if (fix === null) {
      return;
    }
    this.lastManualGpsNodeId = node.node_id;
    this.webSocketService.sendMessageToFlask(JSON.stringify({
      Name: 'Localisation',
      Value: fix,
    }));
  }

  onDestinationChange(event: Event): void {
    const target = event.target as HTMLSelectElement | null;
    this.selectedDestinationId = target?.value ?? '';
  }

  navigateToSelectedDestination(): void {
    if (!this.selectedDestinationId) {
      return;
    }
    this.webSocketService.sendMessageToFlask(JSON.stringify({
      Name: 'NavigationCommand',
      Value: {
        mode: 'go_to',
        destinations: [{ destination_id: this.selectedDestinationId }],
      },
    }));
  }

  private applyMapMetadata(mapMetadata: any): void {
    if (!mapMetadata || typeof mapMetadata !== 'object') {
      return;
    }

    const worldBounds = mapMetadata.world_bounds ?? {};
    const xMin = Number(worldBounds.x_min ?? this.worldBounds.xMin);
    const xMax = Number(worldBounds.x_max ?? this.worldBounds.xMax);
    const yMin = Number(worldBounds.y_min ?? this.worldBounds.yMin);
    const yMax = Number(worldBounds.y_max ?? this.worldBounds.yMax);
    const width = Number(mapMetadata.width_m ?? Math.max(0, xMax - xMin));
    const height = Number(mapMetadata.height_m ?? Math.max(0, yMax - yMin));
    if (width > 0.1) {
      this.trackWidthMeters = width;
    }
    if (height > 0.1) {
      this.trackHeightMeters = height;
    }
    if (Number.isFinite(xMin) && Number.isFinite(xMax) && Number.isFinite(yMin) && Number.isFinite(yMax)) {
      this.worldBounds = {
        xMin,
        xMax,
        yMin,
        yMax,
      };
    }
    this.yAxisInverted = Boolean(mapMetadata.y_axis_inverted ?? true);
  }

  private toMapPercent(point: RoutePoint): RoutePoint {
    const width = Math.max(this.trackWidthMeters, 0.001);
    const height = Math.max(this.trackHeightMeters, 0.001);
    const localX = Number(point.x) - this.worldBounds.xMin;
    const localY = Number(point.y) - this.worldBounds.yMin;
    const percentY = this.yAxisInverted
      ? 100 - (localY * 100) / height
      : (localY * 100) / height;
    return {
      x: Number(((localX * 100) / width).toFixed(3)),
      y: Number(percentY.toFixed(3)),
    };
  }

  private toMapNode(node: any): MapNode | null {
    const nodeId = String(node?.node_id ?? node?.id ?? '');
    if (!nodeId) {
      return null;
    }
    const point = this.toMapPercent({
      x: Number(node?.x ?? 0),
      y: Number(node?.y ?? 0),
    });
    return {
      id: String(node?.id ?? nodeId),
      node_id: nodeId,
      x: Number(node?.x ?? 0),
      y: Number(node?.y ?? 0),
      yaw: Number(node?.yaw ?? 0),
      attr: Number(node?.attr ?? 0),
      is_start: Boolean(node?.is_start),
      label: String(node?.label ?? `Node ${nodeId}`),
      semantic_types: Array.isArray(node?.semantic_types)
        ? node.semantic_types.map((item: any) => String(item))
        : [],
      percentX: point.x,
      percentY: point.y,
    };
  }

  private buildLocalisationFix(node: MapNode): Record<string, unknown> | null {
    const localX = Number(node.x) - this.worldBounds.xMin;
    const localY = Number(node.y) - this.worldBounds.yMin;
    if (!Number.isFinite(localX) || !Number.isFinite(localY)) {
      return null;
    }
    const timestamp = Number((Date.now() / 1000).toFixed(3));
    const posB = this.yAxisInverted
      ? this.trackHeightMeters - localY
      : localY;
    return {
      header: {
        stamp: timestamp,
      },
      timestamp,
      posA: Number(localX.toFixed(6)),
      posB: Number(posB.toFixed(6)),
      rotA: 0.0,
      rotB: 0.0,
      meta: {
        node_id: node.node_id,
        source: 'manual_localisation',
        manual: true,
      },
    };
  }

  public get isAutoMode(): boolean {
    return this.currentDrivingMode === 'auto';
  }

  public get mapHint(): string {
    if (this.isAutoMode) {
      return 'Auto: click en un nodo para fijar GPS manual';
    }
    return 'Click en el mapa para fijar destino';
  }

  public isCurrentNode(node: MapNode): boolean {
    return node.node_id === this.currentNodeId;
  }

  public isUpcomingNode(node: MapNode): boolean {
    return node.node_id === this.upcomingNodeId;
  }

  public isManualGpsNode(node: MapNode): boolean {
    return Boolean(this.lastManualGpsNodeId) && node.node_id === this.lastManualGpsNodeId;
  }

  public nodeRadius(node: MapNode): number {
    if (this.isCurrentNode(node)) {
      return 1.25;
    }
    if (this.isUpcomingNode(node)) {
      return 1.05;
    }
    if (this.isManualGpsNode(node)) {
      return 0.95;
    }
    return 0.6;
  }

  public nodeTooltip(node: MapNode): string {
    const tags = node.semantic_types.length > 0 ? ` · ${node.semantic_types.join(', ')}` : '';
    return `${node.label} (${node.node_id})${tags}`;
  }
}
