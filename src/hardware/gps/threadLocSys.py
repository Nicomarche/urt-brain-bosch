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

Protocolo de competencia (ECC-BFMC Computer):
1. Conectar al TrafficCommunicationServer TCP:5000
2. Enviar  {"reqORinfo": "request", "type": "locsysDevice", "DeviceID": <ID>}
3. Recibir {"response": "<IP>:<PORT>", ...}
4. Conectar al locsys device TCP:<PORT>
5. Recibir {"x": float, "y": float, "yaw_rad"?: float}\n

En simulación (MOTOR_OUTPUT == "zmq"):
Se omite el paso 1-3 y se conecta directamente a SIM_LOCSYS_HOST:SIM_LOCSYS_PORT
donde sim_bridge.py expone el servidor usando el ground truth de Gazebo
transformado al sistema de coordenadas del mapa OSM.

El fix se emite como mensaje Localisation IPC con world_x/world_y y, si está
disponible, yaw_rad/yaw_deg. Así evitamos el flip de imagen y trabajamos
directo en el frame del mapa OSM, sin necesitar conocer height_m.
"""

from __future__ import annotations

import json
import socket
import time

from src.templates.threadwithstop import ThreadWithStop
from src.core.messaging.allMessages import Localisation
from src.core.messaging.messageHandlerSender import messageHandlerSender

# Defaults que se sobreescriben con los valores de config.py si está disponible.
_LOCSYS_PORT        = 4691
_LOCSYS_HOST_COMP   = "192.168.50.11"
_TRAFFIC_COMM_HOST  = "192.168.1.1"
_TRAFFIC_COMM_PORT  = 5000
_LOCSYS_DEVICE_ID   = 1
_SIM_LOCSYS_HOST    = "localhost"
_SIM_LOCSYS_PORT    = 4691
_GPS_RECONNECT_S    = 2.0

try:
    from config import (
        LOCSYS_PORT        as _LOCSYS_PORT,
        LOCSYS_HOST_COMP   as _LOCSYS_HOST_COMP,
        TRAFFIC_COMM_HOST  as _TRAFFIC_COMM_HOST,
        TRAFFIC_COMM_PORT  as _TRAFFIC_COMM_PORT,
        LOCSYS_DEVICE_ID   as _LOCSYS_DEVICE_ID,
        SIM_LOCSYS_HOST    as _SIM_LOCSYS_HOST,
        SIM_LOCSYS_PORT    as _SIM_LOCSYS_PORT,
        GPS_RECONNECT_S    as _GPS_RECONNECT_S,
    )
except ImportError:
    pass

# Socket read timeout (s). Keeps thread_work responsive to stop() while
# waiting for the GPS stream.
_SOCKET_TIMEOUT_S = 1.5


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

    # ------------------------------------------------------------------

    def _resolve_locsys_address(self) -> tuple[str, int]:
        """Devuelve (host, port) del locsys device.

        Sim: directo a localhost:4691.
        Competencia: pregunta al TrafficCommunicationServer por la IP real.
        """
        if self._sim_mode:
            return _SIM_LOCSYS_HOST, _SIM_LOCSYS_PORT

        try:
            with socket.create_connection(
                (_TRAFFIC_COMM_HOST, _TRAFFIC_COMM_PORT), timeout=3.0
            ) as s:
                req = json.dumps({
                    "reqORinfo": "request",
                    "type": "locsysDevice",
                    "DeviceID": _LOCSYS_DEVICE_ID,
                })
                s.sendall(req.encode("utf-8") + b"\n")
                line = s.makefile("r").readline()
                resp = json.loads(line)
                address = str(resp["response"])
                host, port_str = address.rsplit(":", 1)
                return host.strip(), int(port_str.strip())
        except Exception as exc:
            print(
                f"\033[1;97m[ LocSys ] :\033[0m "
                f"\033[1;93mWARN\033[0m - traffic server ({_TRAFFIC_COMM_HOST}:{_TRAFFIC_COMM_PORT}) "
                f"error: {exc} — usando {_LOCSYS_HOST_COMP}:{_LOCSYS_PORT}"
            )
            return _LOCSYS_HOST_COMP, _LOCSYS_PORT

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

    def thread_work(self) -> None:
        """Ciclo de conexión + recepción. Reconecta automáticamente."""
        if self._blocker.is_set():
            return

        host, port = self._resolve_locsys_address()
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
                buffer = b""
                while not self._blocker.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        # Sin datos en _SOCKET_TIMEOUT_S: check stop flag, seguir.
                        continue
                    if not chunk:
                        # Conexión cerrada por el servidor.
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        raw_line, buffer = buffer.split(b"\n", 1)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            self._emit_fix(data)
                        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                            if self.debugger:
                                print(f"\033[1;97m[ LocSys ] :\033[0m parse error: {line!r}")
        except OSError as exc:
            if not self._blocker.is_set():
                print(
                    f"\033[1;97m[ LocSys ] :\033[0m "
                    f"\033[1;93mWARN\033[0m - Conexión perdida ({exc}), "
                    f"reintentando en {_GPS_RECONNECT_S:.0f}s"
                )
                self._blocker.wait(_GPS_RECONNECT_S)
