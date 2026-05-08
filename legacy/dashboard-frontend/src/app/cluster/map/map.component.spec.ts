import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BehaviorSubject, of } from 'rxjs';

import { ClusterService } from '../cluster.service';
import { WebSocketService } from '../../webSocket/web-socket.service';
import { MapComponent } from './map.component';

describe('MapComponent', () => {
  let component: MapComponent;
  let fixture: ComponentFixture<MapComponent>;
  const drivingMode$ = new BehaviorSubject<string>('manual');
  const webSocketServiceStub = {
    receiveLocation: () => of({ value: { x: 4.1697, y: 6.89, yaw: -27.5 } }),
    receiveSemaphores: () => of({ value: { id: 1, x: 0, y: 0, state: 'green' } }),
    receiveNavigationStatus: () => of({
      value: {
        map_metadata: {
          width_m: 20.696,
          height_m: 13.793884,
          image_width_px: 2000,
          image_height_px: 1333,
          meters_per_pixel: 0.010348,
          y_axis_inverted: false,
        },
      },
    }),
    sendMessageToFlask: jasmine.createSpy('sendMessageToFlask'),
    disconnectSocket: jasmine.createSpy('disconnectSocket'),
  };
  const clusterServiceStub = {
    drivingMode$: drivingMode$.asObservable(),
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MapComponent],
      providers: [
        { provide: WebSocketService, useValue: webSocketServiceStub },
        { provide: ClusterService, useValue: clusterServiceStub },
      ],
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(MapComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('uses the ego yaw from Location instead of a UI slider value', () => {
    expect(component.mapPoseTransform).toContain('rotate(62.50 0 0)');
  });
});
