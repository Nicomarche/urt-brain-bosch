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
    receiveLocation: () => of({ value: { x: 0, y: 0 } }),
    receiveSemaphores: () => of({ value: { id: 1, x: 0, y: 0, state: 'green' } }),
    receiveNavigationStatus: () => of({ value: {} }),
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
});
