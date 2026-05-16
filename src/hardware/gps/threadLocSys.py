# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""threadLocSys — cliente BFMC locsys GPS para simulación y competencia real.

Protocolo de competencia (ECC-BFMC Computer), variante locsysDevice:
1. Conectar al TrafficCommunicationServer TCP:5000
2. Enviar  {"reqORinfo": "request", "type": "locsysDevice", "DeviceID": <ID>}
3. Recibir {"response": "<IP>:<PORT>", ...}
4. Conectar al locsys device TCP:<PORT>
5. Recibir {"x": float, "y": float, "yaw_rad"?: float}\n

Variante locIDsub:
1. Conectar al TrafficCommunicationServer TCP:5000
2. Enviar {"reqORinfo": "info", "type": "locIDsub", "locID": <ID>, "freq": 0.25}
3. Recibir {"type": "location", "x": mm, "y": mm, "z": mm}

En simulación (MOTOR_OUTPUT == "zmq"):
por defecto se conecta directo al LoCSys expuesto por `sim_bridge`
(SIM_LOCSYS_HOST:SIM_LOCSYS_PORT). Ese stream ya viene transformado desde
Gazebo al frame `brain_map`; no se debe usar el `locsys_SIM.py` hardcodeado
del repositorio Shared como fuente de GPS del simulador.

El fix se emite como mensaje Localisation IPC con world_x/world_y y, si está
disponible, yaw_rad/yaw_deg. Así evitamos el flip de imagen y trabajamos
directo en el frame del mapa OSM, sin necesitar conocer height_m.
"""

from __future__ import annotations

import json
import math
import socket
import time
from pathlib import Path

from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import CurrentSpeed, Localisation, Location
from src.core.messaging.messageHandlerSender import messageHandlerSender
from src.core.messaging.messageHandlerSubscriber import messageHandlerSubscriber

# Defaults que se sobreescriben con los valores de config.py si está disponible.
_LOCSYS_PORT        = 4691
_LOCSYS_HOST_COMP   = "192.168.50.11"
_TRAFFIC_COMM_HOST  = "auto"
_TRAFFIC_COMM_PORT  = 5000
_TRAFFIC_COMM_AUTODISCOVERY_ENABLED = True
_TRAFFIC_COMM_DISCOVERY_PORT = 9000
_TRAFFIC_COMM_DISCOVERY_TIMEOUT_S = 5.0
_TRAFFIC_COMM_PUBLIC_KEY_PATH = "auto"
_LOCSYS_DEVICE_ID   = 1
_SIM_LOCSYS_HOST    = "localhost"
_SIM_LOCSYS_PORT    = 4691
_GPS_RECONNECT_S    = 2.0
_LOCSYS_USE_TRAFFIC_COMM_SERVER = True
_LOCSYS_DIRECT_FALLBACK_ENABLED = False
_TRAFFIC_COMM_SEND_EGO_DATA = True
_TRAFFIC_COMM_SEND_PERIOD_S = 0.25
_TRAFFIC_COMM_LOCSYS_MODE = "auto"
_TRAFFIC_COMM_LOCSYS_SUB_FREQ = 0.25
_TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE = 0.001

try:
    from config import (
        LOCSYS_PORT        as _LOCSYS_PORT,
        LOCSYS_HOST_COMP   as _LOCSYS_HOST_COMP,
        TRAFFIC_COMM_HOST  as _TRAFFIC_COMM_HOST,
        TRAFFIC_COMM_PORT  as _TRAFFIC_COMM_PORT,
        TRAFFIC_COMM_AUTODISCOVERY_ENABLED as _TRAFFIC_COMM_AUTODISCOVERY_ENABLED,
        TRAFFIC_COMM_DISCOVERY_PORT as _TRAFFIC_COMM_DISCOVERY_PORT,
        TRAFFIC_COMM_DISCOVERY_TIMEOUT_S as _TRAFFIC_COMM_DISCOVERY_TIMEOUT_S,
        TRAFFIC_COMM_PUBLIC_KEY_PATH as _TRAFFIC_COMM_PUBLIC_KEY_PATH,
        LOCSYS_DEVICE_ID   as _LOCSYS_DEVICE_ID,
        SIM_LOCSYS_HOST    as _SIM_LOCSYS_HOST,
        SIM_LOCSYS_PORT    as _SIM_LOCSYS_PORT,
        GPS_RECONNECT_S    as _GPS_RECONNECT_S,
        LOCSYS_USE_TRAFFIC_COMM_SERVER as _LOCSYS_USE_TRAFFIC_COMM_SERVER,
        LOCSYS_DIRECT_FALLBACK_ENABLED as _LOCSYS_DIRECT_FALLBACK_ENABLED,
        TRAFFIC_COMM_SEND_EGO_DATA as _TRAFFIC_COMM_SEND_EGO_DATA,
        TRAFFIC_COMM_SEND_PERIOD_S as _TRAFFIC_COMM_SEND_PERIOD_S,
        TRAFFIC_COMM_LOCSYS_MODE as _TRAFFIC_COMM_LOCSYS_MODE,
        TRAFFIC_COMM_LOCSYS_SUB_FREQ as _TRAFFIC_COMM_LOCSYS_SUB_FREQ,
        TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE as _TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE,
    )
except ImportError:
    pass

# Socket read timeout (s). Keeps thread_work responsive to stop() while
# waiting for the GPS stream.
_SOCKET_TIMEOUT_S = 1.5
_JSON_DECODER = json.JSONDecoder()
_AUTO_HOST_TOKENS = {"", "auto", "autodiscover", "autodiscovery", "discover"}
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class _TrafficRequestNotRecognised(RuntimeError):
    pass


def _extract_json_objects(buffer: str) -> tuple[list[dict], str]:
    """Parse one or more adjacent JSON objects from a stream buffer."""
    objects: list[dict] = []
    while True:
        stripped = buffer.lstrip()
        if not stripped:
            return objects, ""
        try:
            obj, index = _JSON_DECODER.raw_decode(stripped)
        except json.JSONDecodeError:
            first_json = stripped.find("{")
            if first_json > 0:
                buffer = stripped[first_json:]
                continue
            return objects, stripped
        if isinstance(obj, dict):
            objects.append(obj)
        buffer = stripped[index:]


def _is_auto_traffic_host(host: str) -> bool:
    return str(host or "").strip().lower() in _AUTO_HOST_TOKENS


def _traffic_locsys_mode() -> str:
    mode = str(_TRAFFIC_COMM_LOCSYS_MODE or "auto").strip().lower()
    aliases = {
        "": "auto",
        "locsysdevice": "request",
        "locsys_device": "request",
        "address": "request",
        "locidsub": "subscribe",
        "loc_id_sub": "subscribe",
        "subscription": "subscribe",
    }
    return aliases.get(mode, mode)


def _traffic_public_key_candidates() -> list[Path]:
    configured = str(_TRAFFIC_COMM_PUBLIC_KEY_PATH or "").strip()
    if configured and configured.lower() != "auto":
        path = Path(configured).expanduser()
        return [path if path.is_absolute() else _WORKSPACE_ROOT / path]

    key_dir = _WORKSPACE_ROOT / "src" / "data" / "TrafficCommunication" / "useful"
    return [
        key_dir / "publickey_server.pem",
        key_dir / "publickey_server_test.pem",
    ]


def _verify_traffic_broadcast(plaintext: bytes, signature: bytes) -> bool:
    try:
        from src.data.TrafficCommunication.useful import keyDealer
    except Exception:
        return False

    for key_path in _traffic_public_key_candidates():
        try:
            pub_key = keyDealer.load_public_key(str(key_path))
            if keyDealer.verify_data(pub_key, plaintext, signature):
                return True
        except Exception:
            continue
    return False


def _parse_traffic_broadcast(datagram: bytes) -> int | None:
    parts = datagram.split(b"(-.-)", 1)
    if len(parts) != 2:
        return None
    signature, plaintext = parts
    if not _verify_traffic_broadcast(plaintext, signature):
        return None
    try:
        text = plaintext.decode("utf-8", errors="strict").strip()
        prefix = "listening on:"
        if not text.startswith(prefix):
            return None
        return int(text[len(prefix):].strip())
    except (UnicodeDecodeError, ValueError):
        return None


def _extract_locsys_address_from_response(resp: dict) -> str | None:
    for key in ("response", "value", "value1", "address", "locsysDevice"):
        value = resp.get(key)
        if value is not None:
            text = str(value).strip()
            if ":" in text:
                return text
    data = resp.get("data")
    if isinstance(data, dict):
        nested = _extract_locsys_address_from_response(data)
        if nested is not None:
            return nested
    host = resp.get("host", resp.get("ip"))
    port = resp.get("port")
    if host is not None and port is not None:
        return f"{str(host).strip()}:{str(port).strip()}"
    return None


class threadLocSys(ThreadWithStop):
    """Cliente TCP del servidor locsys BFMC.

    thread_work() mantiene la conexión: conecta, lee fixes en loop, y
    reconecta automáticamente tras cualquier fallo de red.
    """

    def __init__(self, queueList, logger, debugger: bool = False):
        # pause=0: el propio thread_work controla los delays de bloqueo
        # (recv con timeout + wait en reconexión).
        super().__init__(pause=0)
        self.queuesList = queueList
        self.logger = logger
        self.debugger = debugger

        try:
            from config import MOTOR_OUTPUT
            self._sim_mode = MOTOR_OUTPUT == "zmq"
        except ImportError:
            self._sim_mode = False

        self.localisationSender = messageHandlerSender(self.queuesList, Localisation)
        self._traffic_location_sub = (
            messageHandlerSubscriber(self.queuesList, Location, "lastOnly", True)
            if _TRAFFIC_COMM_SEND_EGO_DATA
            else None
        )
        self._traffic_speed_sub = (
            messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)
            if _TRAFFIC_COMM_SEND_EGO_DATA
            else None
        )
        self._latest_traffic_location: dict | None = None
        self._latest_traffic_speed_mps: float | None = None
        self._traffic_data_sock: socket.socket | None = None
        self._traffic_next_connect_t = 0.0
        self._traffic_last_send_t = 0.0
        self._traffic_comm_endpoint: tuple[str, int] | None = None
        self._traffic_locsys_runtime_mode: str | None = None

    # ------------------------------------------------------------------

    def _discover_traffic_comm_endpoint(
        self,
        timeout_s: float | None = None,
    ) -> tuple[str, int] | None:
        timeout = (
            float(_TRAFFIC_COMM_DISCOVERY_TIMEOUT_S)
            if timeout_s is None
            else float(timeout_s)
        )
        deadline = time.monotonic() + max(timeout, 0.1)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", int(_TRAFFIC_COMM_DISCOVERY_PORT)))
            except OSError as exc:
                if self.debugger:
                    print(
                        f"\033[1;97m[ LocSys ] :\033[0m "
                        f"traffic autodiscovery bind error on UDP:{_TRAFFIC_COMM_DISCOVERY_PORT}: {exc}"
                    )
                return None
            sock.settimeout(0.25)
            if self.debugger:
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"Buscando TrafficCommunicationServer por UDP:{_TRAFFIC_COMM_DISCOVERY_PORT}"
                )
            while time.monotonic() < deadline and not self._blocker.is_set():
                try:
                    datagram, address = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                port = _parse_traffic_broadcast(datagram)
                if port is None:
                    continue
                host = str(address[0])
                return host, int(port)
        return None

    def _resolve_traffic_comm_endpoint(
        self,
        *,
        discovery_timeout_s: float | None = None,
    ) -> tuple[str, int]:
        if self._traffic_comm_endpoint is not None:
            return self._traffic_comm_endpoint

        auto_host = _is_auto_traffic_host(str(_TRAFFIC_COMM_HOST))
        if auto_host or _TRAFFIC_COMM_AUTODISCOVERY_ENABLED:
            endpoint = self._discover_traffic_comm_endpoint(discovery_timeout_s)
            if endpoint is not None:
                self._traffic_comm_endpoint = endpoint
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;92mINFO\033[0m - TrafficCommunicationServer descubierto en "
                    f"{endpoint[0]}:{endpoint[1]}"
                )
                return endpoint
            if auto_host:
                raise TimeoutError(
                    "TrafficCommunicationServer autodiscovery timed out"
                )

        endpoint = (str(_TRAFFIC_COMM_HOST), int(_TRAFFIC_COMM_PORT))
        self._traffic_comm_endpoint = endpoint
        return endpoint

    def _resolve_locsys_address(self) -> tuple[str, int]:
        """Devuelve (host, port) del locsys device.

        Sim: directo al LoCSys del sim_bridge.
        Competencia: pregunta al TrafficCommunicationServer por la IP real.
        """
        if self._sim_mode and not _LOCSYS_USE_TRAFFIC_COMM_SERVER:
            return _SIM_LOCSYS_HOST, _SIM_LOCSYS_PORT
        if not _LOCSYS_USE_TRAFFIC_COMM_SERVER:
            return _LOCSYS_HOST_COMP, _LOCSYS_PORT
        if _traffic_locsys_mode() == "subscribe":
            raise RuntimeError("TRAFFIC_COMM_LOCSYS_MODE=subscribe no usa locsys device address")

        try:
            traffic_host, traffic_port = self._resolve_traffic_comm_endpoint()
            with socket.create_connection(
                (traffic_host, traffic_port), timeout=3.0
            ) as s:
                s.settimeout(3.0)
                req = json.dumps({
                    "reqORinfo": "request",
                    "type": "locsysDevice",
                    "DeviceID": _LOCSYS_DEVICE_ID,
                })
                s.sendall(req.encode("utf-8") + b"\n")
                buffer = ""
                deadline = time.monotonic() + 3.0
                resp = None
                while time.monotonic() < deadline and resp is None:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    objects, buffer = _extract_json_objects(buffer)
                    if objects:
                        resp = objects[0]
                if resp is None:
                    raise TimeoutError("traffic server did not return a JSON response")
                if resp.get("error") is not None:
                    message = (
                        f"traffic server locsysDevice DeviceID={_LOCSYS_DEVICE_ID} "
                        f"error: {resp.get('error')} response={resp!r}"
                    )
                    if "request not recognised" in str(resp.get("error")).lower():
                        raise _TrafficRequestNotRecognised(message)
                    raise ValueError(message)
                address = _extract_locsys_address_from_response(resp)
                if address is None:
                    raise ValueError(f"traffic server returned no locsys address: {resp!r}")
                host, port_str = address.rsplit(":", 1)
                return host.strip(), int(port_str.strip())
        except _TrafficRequestNotRecognised:
            raise
        except Exception as exc:
            endpoint_label = (
                "auto"
                if _is_auto_traffic_host(str(_TRAFFIC_COMM_HOST))
                else f"{_TRAFFIC_COMM_HOST}:{_TRAFFIC_COMM_PORT}"
            )
            fallback_host, fallback_port = (
                (_SIM_LOCSYS_HOST, _SIM_LOCSYS_PORT)
                if self._sim_mode
                else (_LOCSYS_HOST_COMP, _LOCSYS_PORT)
            )
            if not _LOCSYS_DIRECT_FALLBACK_ENABLED:
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - traffic server ({endpoint_label}) "
                    f"error: {exc}; sin fallback directo a {fallback_host}:{fallback_port}. "
                    f"Reintentando discovery."
                )
                raise
            print(
                f"\033[1;97m[ LocSys ] :\033[0m "
                f"\033[1;93mWARN\033[0m - traffic server ({endpoint_label}) "
                f"error: {exc} — usando fallback directo {fallback_host}:{fallback_port}"
            )
            return fallback_host, fallback_port

    # ------------------------------------------------------------------

    @staticmethod
    def _as_float(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _receive_traffic_inputs(self) -> None:
        if self._traffic_location_sub is not None:
            location = self._traffic_location_sub.receive()
            if isinstance(location, dict):
                self._latest_traffic_location = location

        if self._traffic_speed_sub is not None:
            speed_raw = self._traffic_speed_sub.receive()
            speed = self._as_float(speed_raw)
            if speed is not None:
                self._latest_traffic_speed_mps = speed * 0.001

    def _traffic_socket(self, now: float) -> socket.socket | None:
        if self._traffic_data_sock is not None:
            return self._traffic_data_sock
        if now < self._traffic_next_connect_t:
            return None
        try:
            traffic_host, traffic_port = self._resolve_traffic_comm_endpoint(
                discovery_timeout_s=min(float(_TRAFFIC_COMM_DISCOVERY_TIMEOUT_S), 1.0)
            )
            sock = socket.create_connection(
                (traffic_host, traffic_port), timeout=0.5
            )
            sock.settimeout(0.5)
            self._traffic_data_sock = sock
            return sock
        except OSError:
            self._traffic_next_connect_t = now + float(_GPS_RECONNECT_S)
            return None

    def _close_traffic_socket(self) -> None:
        sock = self._traffic_data_sock
        self._traffic_data_sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _send_ego_data_to_traffic_server(self) -> None:
        if not _TRAFFIC_COMM_SEND_EGO_DATA:
            return

        self._receive_traffic_inputs()
        now = time.monotonic()
        if now - self._traffic_last_send_t < float(_TRAFFIC_COMM_SEND_PERIOD_S):
            return

        location = self._latest_traffic_location
        if not isinstance(location, dict):
            return

        x = self._as_float(location.get("x", location.get("world_x")))
        y = self._as_float(location.get("y", location.get("world_y")))
        if x is None or y is None:
            return

        yaw = self._as_float(location.get("yaw_rad"))
        if yaw is None:
            yaw_deg = self._as_float(location.get("yaw", location.get("yaw_deg")))
            yaw = math.radians(yaw_deg) if yaw_deg is not None else 0.0

        speed = self._latest_traffic_speed_mps
        if speed is None:
            speed = 0.0

        messages = [
            {"reqORinfo": "info", "type": "devicePos", "value1": x, "value2": y},
            {"reqORinfo": "info", "type": "deviceRot", "value1": yaw},
            {"reqORinfo": "info", "type": "deviceSpeed", "value1": speed},
        ]
        payload = "".join(json.dumps(msg) for msg in messages).encode("utf-8")
        sock = self._traffic_socket(now)
        if sock is None:
            return

        try:
            sock.sendall(payload)
            self._traffic_last_send_t = now
        except OSError:
            self._close_traffic_socket()
            self._traffic_next_connect_t = now + float(_GPS_RECONNECT_S)

    # ------------------------------------------------------------------

    def _emit_fix(self, data: dict) -> None:
        """Envía un Localisation IPC con coordenadas en el frame del OSM."""
        x = float(data["x"])
        y = float(data["y"])
        payload = {
            "timestamp": time.time(),
            # world_x/world_y bypasean la conversión posA/posB con y_axis_inverted
            # en localisation_to_world_pose del route handler OSM. El pose
            # estimator recibe coordenadas directas del mapa sin necesitar
            # height_m.
            "world_x": x,
            "world_y": y,
            "posA": x,   # para compatibilidad con el dashboard
            "posB": y,
            "rotA": 0.0,
            "rotB": 0.0,
            "meta": {
                "source": "gps_localisation",
                "manual": False,
            },
        }
        try:
            if data.get("yaw_rad") is not None:
                payload["yaw_rad"] = float(data["yaw_rad"])
            if data.get("yaw_deg") is not None:
                payload["yaw_deg"] = float(data["yaw_deg"])
        except (TypeError, ValueError):
            pass
        self.localisationSender.send(payload)
        if self.debugger:
            yaw_dbg = payload.get("yaw_deg")
            if yaw_dbg is None and payload.get("yaw_rad") is not None:
                yaw_dbg = float(payload["yaw_rad"]) * 180.0 / 3.141592653589793
            if yaw_dbg is None:
                print(f"\033[1;97m[ LocSys ] :\033[0m GPS fix x={x:.3f} y={y:.3f}")
            else:
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"GPS fix x={x:.3f} y={y:.3f} yaw={float(yaw_dbg):+.1f}°"
                )

    # ------------------------------------------------------------------

    def _traffic_subscription_fix_from_message(self, data: dict) -> dict | None:
        if data.get("error") is not None:
            raise ValueError(f"traffic locIDsub error: {data.get('error')} response={data!r}")

        nested = data.get("location", data.get("data"))
        if isinstance(nested, dict):
            nested_fix = self._traffic_subscription_fix_from_message(nested)
            if nested_fix is not None:
                return nested_fix

        msg_type = str(data.get("type", "location")).strip().lower()
        if msg_type not in {"location", "locsys", "locsyslocation"}:
            return None

        candidates = (
            ("x", "y"),
            ("world_x", "world_y"),
            ("posA", "posB"),
            ("value1", "value2"),
        )
        x = y = None
        for x_key, y_key in candidates:
            x = self._as_float(data.get(x_key))
            y = self._as_float(data.get(y_key))
            if x is not None and y is not None:
                break
        if x is None or y is None:
            return None

        scale = float(_TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE)
        fix = {
            "x": x * scale,
            "y": y * scale,
        }

        yaw_rad = self._as_float(data.get("yaw_rad"))
        if yaw_rad is not None:
            fix["yaw_rad"] = yaw_rad
        else:
            yaw_deg = self._as_float(data.get("yaw_deg", data.get("yaw")))
            if yaw_deg is not None:
                fix["yaw_deg"] = yaw_deg
        return fix

    def _thread_work_traffic_subscription(self) -> None:
        """Recibe GPS directo desde TrafficCommunicationServer usando locIDsub."""
        try:
            traffic_host, traffic_port = self._resolve_traffic_comm_endpoint()
        except Exception as exc:
            if not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - No se pudo resolver "
                    f"TrafficCommunicationServer para locIDsub ({exc}); reintentando en "
                    f"{_GPS_RECONNECT_S:.0f}s"
                )
                self._blocker.wait(_GPS_RECONNECT_S)
            return

        print(
            f"\033[1;97m[ LocSys ] :\033[0m "
            f"\033[1;92mINFO\033[0m - Suscribiendo GPS locIDsub "
            f"id={_LOCSYS_DEVICE_ID} en {traffic_host}:{traffic_port}"
        )

        try:
            with socket.create_connection((traffic_host, traffic_port), timeout=5.0) as sock:
                sock.settimeout(_SOCKET_TIMEOUT_S)
                req = {
                    "reqORinfo": "info",
                    "type": "locIDsub",
                    "locID": _LOCSYS_DEVICE_ID,
                    "freq": float(_TRAFFIC_COMM_LOCSYS_SUB_FREQ),
                }
                sock.sendall(json.dumps(req).encode("utf-8"))
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;92mINFO\033[0m - Conectado al stream GPS locIDsub"
                )
                buffer = ""
                while not self._blocker.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        self._send_ego_data_to_traffic_server()
                        continue
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    objects, buffer = _extract_json_objects(buffer)
                    for data in objects:
                        try:
                            fix = self._traffic_subscription_fix_from_message(data)
                            if fix is not None:
                                self._emit_fix(fix)
                            elif self.debugger:
                                print(
                                    f"\033[1;97m[ LocSys ] :\033[0m "
                                    f"traffic locIDsub ignored: {data!r}"
                                )
                        except TypeError as exc:
                            if self.debugger:
                                print(
                                    f"\033[1;97m[ LocSys ] :\033[0m "
                                    f"traffic locIDsub parse error: {exc}"
                                )
                    self._send_ego_data_to_traffic_server()
        except Exception as exc:
            if not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - Stream GPS locIDsub perdido ({exc}), "
                    f"reintentando en {_GPS_RECONNECT_S:.0f}s"
                )
                self._blocker.wait(_GPS_RECONNECT_S)

    def thread_work(self) -> None:
        """Ciclo de conexión + recepción. Reconecta automáticamente."""
        if self._blocker.is_set():
            return

        mode = self._traffic_locsys_runtime_mode or _traffic_locsys_mode()
        if (
            mode == "subscribe"
            and _LOCSYS_USE_TRAFFIC_COMM_SERVER
            and not (self._sim_mode and not _LOCSYS_USE_TRAFFIC_COMM_SERVER)
        ):
            self._thread_work_traffic_subscription()
            return

        try:
            host, port = self._resolve_locsys_address()
        except _TrafficRequestNotRecognised as exc:
            if mode == "auto" and not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - {exc}; el server descubierto "
                    f"usa el protocolo locIDsub. Probando suscripcion directa."
                )
                self._traffic_locsys_runtime_mode = "subscribe"
                self._thread_work_traffic_subscription()
            elif not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - TrafficCommunicationServer no "
                    f"reconoce locsysDevice ({exc}); reintentando en "
                    f"{_GPS_RECONNECT_S:.0f}s"
                )
                self._blocker.wait(_GPS_RECONNECT_S)
            return
        except Exception as exc:
            if not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - No se pudo resolver locsys vía "
                    f"TrafficCommunicationServer ({exc}); reintentando en "
                    f"{_GPS_RECONNECT_S:.0f}s"
                )
                self._blocker.wait(_GPS_RECONNECT_S)
            return
        print(
            f"\033[1;97m[ LocSys ] :\033[0m "
            f"\033[1;92mINFO\033[0m - Conectando a {host}:{port}"
        )

        try:
            with socket.create_connection((host, port), timeout=5.0) as sock:
                sock.settimeout(_SOCKET_TIMEOUT_S)
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;92mINFO\033[0m - Conectado al servidor locsys {host}:{port}"
                )
                buffer = ""
                while not self._blocker.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        # Sin datos en _SOCKET_TIMEOUT_S: check stop flag, seguir.
                        self._send_ego_data_to_traffic_server()
                        continue
                    if not chunk:
                        # Conexión cerrada por el servidor.
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    objects, buffer = _extract_json_objects(buffer)
                    for data in objects:
                        try:
                            self._emit_fix(data)
                        except (KeyError, ValueError, TypeError):
                            if self.debugger:
                                print(f"\033[1;97m[ LocSys ] :\033[0m parse error: {data!r}")
                    self._send_ego_data_to_traffic_server()
        except OSError as exc:
            if not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - Conexión perdida ({exc}), "
                    f"reintentando en {_GPS_RECONNECT_S:.0f}s"
                )
                self._blocker.wait(_GPS_RECONNECT_S)

    def stop(self) -> None:
        self._close_traffic_socket()
        super().stop()
