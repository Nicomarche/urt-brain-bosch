# Project Plan - BFMC Brain (URT)

## General Information

- **Project:** Bosch Future Mobility Challenge - Vehicle Control System
- **Team:** URT (Universidad Rafael Urdaneta)
- **Platform:** Raspberry Pi + STM32 Nucleo
- **Repository:** urt-brain-bosch

---

## 1. Project Description

Autonomous control system for a scale vehicle participating in the BFMC competition. The system includes:

- Movement control (speed and steering)
- Image processing with camera
- Automatic line following
- Web dashboard for monitoring and remote control
- Communication with traffic and semaphore servers
- Multiple operation modes (Manual, Autonomous, Legacy)

---

## 2. System Architecture

### 2.1 Hardware Components
- Raspberry Pi 4/5 (main processing)
- STM32 Nucleo (motor control)
- Pi Camera (vision)
- IMU (orientation)
- DC Motors (traction and steering)

### 2.2 Software Modules

| Module | Location | Function |
|--------|----------|----------|
| Main | `main.py` | Main process orchestrator |
| Camera | `src/hardware/camera/` | Image capture and processing |
| Serial Handler | `src/hardware/serialhandler/` | Communication with Nucleo |
| Gateway | `src/gateway/` | Message routing between processes |
| State Machine | `src/statemachine/` | System state control |
| Dashboard | `src/dashboard/` | Angular web interface |
| Traffic Comm | `src/data/TrafficCommunication/` | Traffic server communication |
| Semaphores | `src/data/Semaphores/` | Semaphore state reception |

---

## 3. Implemented Features

### 3.1 Completed

- [x] Base project structure (thread and process templates)
- [x] Serial communication with Nucleo (read/write)
- [x] Camera capture and streaming
- [x] Web dashboard with Angular
- [x] Manual speed and steering control
- [x] State system (STOP, MANUAL, AUTO, LEGACY)
- [x] Basic line following with OpenCV
- [x] Perspective transformation (Bird's Eye View)
- [x] Sliding Window algorithm for lane detection
- [x] PID control for steering
- [x] Parameter adjustment from dashboard
- [x] Autostart services for Raspberry Pi
- [x] Automatic WiFi fallback

### 3.2 In Progress

- [ ] Fine-tuning line following on curves
- [ ] Lookahead parameter optimization
- [ ] HSV calibration for different lighting conditions

### 3.3 Pending

- [ ] Traffic sign detection
- [ ] Traffic light detection
- [ ] Obstacle and pedestrian detection
- [ ] Intersection detection
- [ ] Route planning
- [ ] Localization server integration
- [ ] Automatic parking
- [ ] Roundabout handling

---

## 4. Project Phases

### Phase 1: Base Infrastructure (COMPLETED)
- Raspberry Pi setup
- Functional serial communication
- Basic dashboard operational
- Operation modes implemented

### Phase 2: Vision and Line Following (IN PROGRESS)
- Camera capture and processing
- White and yellow line detection
- Perspective transformation
- Automatic steering control
- Speed adjustment on curves

### Phase 3: Object Detection (NEXT)
- Traffic sign detection
- Traffic light recognition
- Obstacle detection
- Pedestrian detection

### Phase 4: Advanced Navigation (FUTURE)
- Localization server integration
- Route planning
- Intersection handling
- Automatic parking

### Phase 5: Integration and Testing (FUTURE)
- Full track testing
- Performance optimization
- Edge case handling
- Competition preparation

---

## 5. Current System Parameters

### 5.1 Line Following

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_steering | 25° | Maximum steering angle |
| steering_sensitivity | 3.0 | Response sensitivity |
| kp | 4.0 | Proportional gain |
| kd | 0.05 | Derivative gain |
| smoothing_factor | 0.3 | Steering smoothing |
| lookahead | 0.4 | Curve anticipation |
| base_speed | 10 | Base speed |
| min_speed | 5 | Speed on curves |

### 5.2 ROI (Region of Interest)

| Parameter | Value | Description |
|-----------|-------|-------------|
| roi_height_start | 0.55 | ROI vertical start |
| roi_height_end | 0.92 | ROI vertical end |
| roi_width_margin_top | 0.35 | Top lateral margin |
| roi_width_margin_bottom | 0.15 | Bottom lateral margin |

### 5.3 Color Detection (HSV)

**White Lines:**
- H: 81-180
- S: 0-98
- V: 200-255

**Yellow Lines:**
- H: 173-86
- S: 100-255
- V: 100-255

---

## 6. Known Issues and Solutions

### 6.1 Line Following

| Issue | Cause | Solution |
|-------|-------|----------|
| Late turning on curves | Low lookahead | Increase lookahead to 0.4+ |
| Excessive oscillation | Smoothing too low | Adjust smoothing_factor |
| Lines not detected | HSV miscalibrated | Recalibrate for lighting |
| Incorrect pink point | Untransformed coordinates | Apply inverse transformation |

### 6.2 General System

| Issue | Cause | Solution |
|-------|-------|----------|
| Dashboard won't connect | WebSocket down | Restart brain service |
| Camera not responding | Blocked process | Check camera permissions |
| Serial timeout | Nucleo disconnected | Check USB connection |

---

## 7. Key File Structure

```
urt-brain-bosch/
├── main.py                          # Main entry point
├── src/
│   ├── hardware/
│   │   ├── camera/
│   │   │   ├── processCamera.py     # Camera process
│   │   │   └── threads/
│   │   │       ├── threadCamera.py  # Frame capture
│   │   │       └── threadLineFollowing.py  # Line following
│   │   └── serialhandler/
│   │       ├── processSerialHandler.py
│   │       └── threads/
│   │           ├── threadRead.py    # Serial read
│   │           └── threadWrite.py   # Serial write
│   ├── statemachine/
│   │   ├── stateMachine.py          # State machine
│   │   └── systemMode.py            # Mode definitions
│   ├── gateway/
│   │   └── processGateway.py        # Message router
│   └── dashboard/
│       └── frontend/                # Angular app
├── services/                        # systemd services
└── monitoring/
    └── project-plan.md              # This file
```

---

## 8. Useful Commands

### Start the system
```bash
cd /home/pi/Documents/urt-brain-bosch
python3 main.py
```

### Start dashboard
```bash
cd src/dashboard/frontend
ng serve --host 0.0.0.0
```

### View logs
```bash
journalctl -u brain-monitor -f
journalctl -u angular-dashboard -f
```

### Restart services
```bash
sudo systemctl restart brain-monitor
sudo systemctl restart angular-dashboard
```

---

## 9. Contact and Resources

- **BFMC Documentation:** https://bosch-future-mobility-challenge-documentation.readthedocs-hosted.com/
- **Repository:** /home/pi/Documents/urt-brain-bosch

---

## 10. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-02-02 | Fixed steering angle calculation | - |
| 2026-02-02 | Adjusted lookahead for curve anticipation | - |
| 2026-02-02 | Limited steering to 25° | - |
| 2026-02-02 | Created project plan | - |

---

*Last updated: February 2, 2026*
