# Odometria y Dead Reckoning - descripcion tecnica

## 1. Sensor fisico

- Modelo Hall: A49E. Pin: A1 del Nucleo F401RE.
- Tension idle: ~1.67 V (`read_u16 ~= 33000`).
- Disparo del pulso: sample < 28000 (~1.41 V); rearmado a sample > 30000 (~1.51 V).
- 1 iman por rueda -> 1 pulso por revolucion.
- Distancia por pulso: `pi * diametro = pi * 65 mm = 204.2 mm`.

## 2. Firmware Nucleo

Fork: `Nicomarche/urt-cerebro`, branch `master`.

- `include/periodics/hallspeed.hpp`: `HALL_PULSES_PER_REV` define cuantos pulsos hay por revolucion.
- Cadencia de publicacion: 50 Hz.
- `@speed`: promedio movil de los ultimos 4 pulsos (mm/s). Si no hay pulso por >300 ms publica `@speed:0`.
- `@odo`: `pulseCount * 204.2` como `uint32` en mm. No acumula drift de float.
- `#odoreset:<x>;;` resetea `pulseCount=0` y responde `@odoreset:1;;`.
- Requiere `kl=15` o `kl=30`; `#hallspeed:1;;` activa la publicacion periodica.

## 3. Protocolo serial brain-Nucleo

- Baud: 115200 8N1, terminaciones `\r\n`.
- Formato: `@key:payload;;` para salida del Nucleo; `#key:payload;;` para comandos del brain.
- Parser del brain: `src/hardware/serialhandler/threads/threadRead.py:145` hace `split(":", 1)` y normaliza la accion con `re.sub`.
- `@speed`: float, mm/s, sin signo. El Hall no distingue sentido.
- `@odo`: uint32, mm acumulados desde el ultimo reset.
- `@steer`: float, decimas de grado. En wire legacy, positivo significa derecha.
- `@imu`: dict/string con `roll`, `pitch`, `yaw`, `accelx`, `accely`, `accelz`.
- Implementacion nueva: `OdoDistance` esta en `src/core/messaging/allMessages.py:385`; el parser `@odo` esta en `src/hardware/serialhandler/threads/threadRead.py:180`.

## 4. Brain - pipeline IPC

1. `threadRead` parsea `@odo` y publica `OdoDistance` (`int`, mm acumulados).
2. `threadSimFeedback` recibe `odo_mm` del feedback ZMQ del simulador y publica el mismo `OdoDistance`; ver `src/hardware/serialhandler/threads/threadSimFeedback.py:182`.
3. `threadDeadReckoningPure` consume `OdoDistance`, `CurrentSteer`, `ImuData` y `Location`; ver `src/localization/dead_reckoning_pure_thread.py:56`.
4. Publica `DeadReckoningPose` (`dict {"x", "y", "yaw_deg", "timestamp", "anchored"}`), declarado en `src/core/messaging/allMessages.py:401`.
5. `processDashboard` lo recibe y emite SocketIO `"DeadReckoning"`; ver `src/dashboard/processDashboard.py:344` y `src/dashboard/processDashboard.py:726`.
6. El cliente Qt define `EVT_DEAD_RECKONING` en `src/dashboard/gui/client/events.py:74`, expone `dead_reckoning_signal` en `src/dashboard/gui/client/socketio_client.py:172` y actualiza `_DrDot` en `src/dashboard/gui/widgets/map_view.py:465`.

## 5. Modelo cinematico

Bicycle por distancia, no por tiempo:

```text
delta_d = delta(@odo) / 1000                         [m]
yaw_efectivo = imu_yaw_abs + yaw_offset_anchor
delta_yaw = -(delta_d / L) * tan(steer)              [rad]
yaw_mid = yaw_efectivo + 0.5 * delta_yaw
delta_x = delta_d * cos(yaw_mid)
delta_y = delta_d * sin(yaw_mid)
x += delta_x
y += delta_y
```

La convencion del frame OSM/dashboard es `x` a la derecha, `y` hacia abajo y yaw positivo horario. Por eso el termino de yaw lleva signo negativo para un giro fisico a la izquierda. La conversion de steer replica `relocalization_thread`: wire `+` derecha -> matematico `+` izquierda.

El thread carga dos constantes desde `config.py:402` y `config.py:417`: `TRACKING_STEER_SIGN_DR` (default `1.0`) ajusta el signo del feedback de steering por instalacion del servo; `TRACKING_IMU_YAW_SIGN` (`1.0` en sim, `-1.0` en hardware) normaliza el yaw IMU al frame OSM clockwise-positive. Cambiarlas requiere revalidar contra `relocalization_thread._parse_steer_rad` y `pose_estimator`, que usan las mismas constantes y la misma convencion.

Wheelbase: `L = 0.260 m`, el mismo valor base de `src/localization/dead_reckoning.py:38`.

## 6. Anclaje

- Al primer `Location` IPC con `meta.source in {"ego_pose_pose_estimator", "ego_pose_relocalization"}`:
  - `x, y <- payload["x"], payload["y"]` en frame OSM.
  - Si el payload trae `yaw_rad`: `yaw_offset = location_yaw - imu_yaw_del_momento`.
  - Si no trae `yaw_rad`, se usa `yaw`/`yaw_deg` como fallback; si tampoco hay yaw, `yaw_offset = 0` y el yaw queda dominado por el IMU absoluto.
  - `anchored = True`; `_last_odo_mm = None` para que el siguiente delta empiece desde el anchor.
- Se requiere haber recibido al menos un `ImuData` con `imu_age < 1.0 s` antes de anclar. Sin eso, el offset podria calcularse contra `yaw=0` (valor inicial) o contra un heading viejo, introduciendo un sesgo permanente. El mismo umbral de 1 s se usa despues del anchor para el guard de IMU stale.
- Antes del anchor, el thread publica solo `{"anchored": false}` de forma espaciada; el dashboard mantiene oculto el marcador verde.
- El evento `dead_reckoning.dr_anchored` se emite una sola vez desde `src/localization/dead_reckoning_pure_thread.py:184`.
- Por que anclar al `Location` y no al GPS crudo: asi el dot verde arranca en la misma posicion que el cursor rojo (`_CarCursor`), que tambien sigue la pose ego del pose_estimator/relocalization. El GPS azul queda como medicion cruda independiente; la divergencia posterior entre verde y azul sigue mostrando el error acumulado del DR frente a LoCSys.

## 7. Reset de odometro

- El brain no envia `#odoreset`.
- El thread usa deltas entre muestras consecutivas: `delta(@odo)`.
- Reinicio del brain en marcha: `_last_odo_mm` se inicializa con la primera muestra post-reinicio, sin salto de posicion.
- Overflow o reset externo del `uint32`: un delta negativo se descarta y se actualiza la referencia para continuar con la muestra siguiente.

## 8. Simulador (`MOTOR_OUTPUT=zmq`)

Repo: `/Users/luciogarcia/urt-simulator`.

- `Simulator/src/models_pkg/prius_rccar/model.sdf:693`: `JointStatePublisher` preexistente, fuente de `axis1.position` y `axis1.velocity`.
- `Simulator/src/models_pkg/prius_rccar/model.sdf:707`: `OdometryPublisher` nuevo, publica `/model/automobile/odometry` como verdad independiente de diagnostico.
- `sim_bridge.py:332`: estado acumulado `_odo_m_acum` y `_last_rear_wheel_pos_rad`.
- `sim_bridge.py:633`: `_on_joint_state` promedia posiciones angulares de ruedas traseras y suma `abs(delta_rad) * WHEEL_RADIUS_M`.
- `sim_bridge.py:1250` y `sim_bridge.py:1264`: el payload feedback agrega `"odo_mm": int(round(_odo_m_acum * 1000))`.
- El servidor LoCSys del `sim_bridge` espera un `set_pose` fresco antes de emitir GPS a cada cliente nuevo, para no filtrar como primer fix una pose cacheada de un run anterior.

Mapping resultante:

- Nucleo real: `@odo = pulseCount * 204.2 mm = rotaciones * 2pi * 0.0325 m * 1000`.
- Gazebo: `odo_mm = integral |delta(axis1.position_rear)| * 0.031265 m * 1000`.
- Conceptualmente ambos son rotacion de rueda por radio. El valor exacto del radio puede diferir entre hardware y SDF, pero la integracion de DR solo consume el delta publicado.

## 9. Logs JSONL

El logger comun es `src/utils/live_log.py`. El `campaign_runner` lee `temp/logs/<run>/brain.jsonl`.

- `locsys.gps_fix`: `world_x`, `world_y`, `yaw_rad`, `source`, `sim_mode`; emitido en `src/hardware/gps/threadLocSys.py:538`.
- `pose_estimator.pose_published`: `fused_x/y/yaw_rad`, `raw_x/y/yaw_rad`, `speed_mps`, `steer_rad`, `reloc_*`; ya existia en `src/localization/pose_estimator_thread.py:882`.
- `dead_reckoning.dr_pose`: `x`, `y`, `yaw_rad`, `delta_d_m`, `odo_mm`, `steer_rad`, `imu_yaw_rad`, `anchored`; emitido en `src/localization/dead_reckoning_pure_thread.py:226`.
- `dead_reckoning.dr_anchored`: `anchor_x`, `anchor_y`, `anchor_yaw_rad`, `anchor_source`, `imu_yaw_rad`, `yaw_offset_rad`.
- `dead_reckoning.dr_imu_stale`: emitido una sola vez por episodio de IMU stale (>1 s sin sample fresco) desde `src/localization/dead_reckoning_pure_thread.py:249`; incluye `imu_age_s`. Cuando el IMU vuelve, el thread reanuda `dr_pose` sin re-anchor porque el offset sigue siendo valido.

## 10. Overlay PNG

- `overlay.png`: expected vs ground truth. No cambia.
- `overlay_dr.png`: compara ground truth gris, GPS azul, pose fusionado rojo y DR puro verde.
- Recoleccion de paths: `tools/dev/campaign_runner.py:843`.
- Render del overlay DR: `tools/dev/campaign_runner.py:1842`.
- Registro del artefacto: `tools/dev/campaign_runner.py:2415`.

## 11. Limitaciones conocidas

- `delta_yaw` del bicycle model es ideal; derrape en curvas rapidas introduce error que no se modela.
- El yaw absoluto depende de la calibracion del BNO055 o del yaw sintetico del simulador. Si el yaw esta rotado, todo el DR queda rotado.
- El anchor se hace una sola vez. No hay re-anchor periodico por GPS.
- Sin IMU fresco, el thread despublica la pose anclada con `anchored=false`; el dot del dashboard se oculta. No hay fallback de integracion por steer puro.
- Sin `Location` ego, no se ancla nunca; el dot no aparece en el mapa.
- En sim, `odo_mm` depende de la fidelidad del `JointStatePublisher` y del contacto de ruedas de Gazebo. Como en el auto real, el odometro mide rotacion de rueda, no desplazamiento real si hay resbalamiento.
