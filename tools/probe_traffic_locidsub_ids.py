#!/usr/bin/env python3
"""Probe BFMC TrafficCommunicationServer locIDsub IDs.

Default behavior:
- Discover TrafficCommunicationServer through its signed UDP broadcast.
- Try locIDsub for IDs 1..22.
- Mark an ID active when a location JSON arrives before the per-ID timeout.

This script is diagnostic-only. It does not modify the server.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JSON_DECODER = json.JSONDecoder()


def _config_value(name: str, default: Any) -> Any:
    try:
        import config  # type: ignore
    except Exception:
        return default
    return getattr(config, name, default)


def _parse_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(token))
    return list(dict.fromkeys(ids))


def _extract_json_objects(buffer: str) -> tuple[list[dict[str, Any]], str]:
    objects: list[dict[str, Any]] = []
    while True:
        stripped = buffer.lstrip()
        if not stripped:
            return objects, ""
        try:
            obj, index = JSON_DECODER.raw_decode(stripped)
        except json.JSONDecodeError:
            first_json = stripped.find("{")
            if first_json > 0:
                buffer = stripped[first_json:]
                continue
            return objects, stripped
        if isinstance(obj, dict):
            objects.append(obj)
        buffer = stripped[index:]


def _public_key_candidates(configured: str | None) -> list[Path]:
    configured = (configured or "").strip()
    if configured and configured.lower() != "auto":
        path = Path(configured).expanduser()
        return [path if path.is_absolute() else REPO_ROOT / path]

    return [
        REPO_ROOT / "src" / "data" / "TrafficCommunication" / "useful" / "publickey_server.pem",
        REPO_ROOT / "src" / "data" / "TrafficCommunication" / "useful" / "publickey_server_test.pem",
        Path.home()
        / "trafficComunication"
        / "trafficCommunicationServer"
        / "Useful"
        / "publickey_server.pem",
        Path.home()
        / "trafficComunication"
        / "trafficCommunicationServer"
        / "Useful"
        / "publickey_server_test.pem",
    ]


def _verify_traffic_broadcast(
    plaintext: bytes,
    signature: bytes,
    key_paths: list[Path],
) -> bool:
    try:
        from src.data.TrafficCommunication.useful import keyDealer
    except Exception:
        return False

    for key_path in key_paths:
        if not key_path.exists():
            continue
        try:
            pub_key = keyDealer.load_public_key(str(key_path))
            if keyDealer.verify_data(pub_key, plaintext, signature):
                return True
        except Exception:
            continue
    return False


def _parse_traffic_broadcast(
    datagram: bytes,
    key_paths: list[Path],
    allow_unsigned: bool,
) -> int | None:
    parts = datagram.split(b"(-.-)", 1)
    if len(parts) == 2:
        signature, plaintext = parts
        if not allow_unsigned and not _verify_traffic_broadcast(
            plaintext,
            signature,
            key_paths,
        ):
            return None
    elif allow_unsigned:
        plaintext = datagram
    else:
        return None

    try:
        text = plaintext.decode("utf-8", errors="strict").strip()
        prefix = "listening on:"
        if not text.startswith(prefix):
            return None
        return int(text[len(prefix) :].strip())
    except (UnicodeDecodeError, ValueError):
        return None


def discover_traffic_server(
    discovery_port: int,
    timeout_s: float,
    public_key_path: str,
    allow_unsigned: bool,
) -> tuple[str, int]:
    key_paths = _public_key_candidates(public_key_path)
    deadline = time.monotonic() + max(timeout_s, 0.1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.bind(("", int(discovery_port)))
        sock.settimeout(0.25)
        while time.monotonic() < deadline:
            try:
                datagram, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            port = _parse_traffic_broadcast(datagram, key_paths, allow_unsigned)
            if port is not None:
                return str(address[0]), int(port)
    raise TimeoutError(
        f"TrafficCommunicationServer autodiscovery timed out on UDP:{discovery_port}"
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _location_like(data: dict[str, Any]) -> dict[str, Any] | None:
    nested = data.get("location", data.get("data"))
    if isinstance(nested, dict):
        nested_location = _location_like(nested)
        if nested_location is not None:
            return nested_location

    msg_type = str(data.get("type", "location")).strip().lower()
    if msg_type != "location":
        return None

    x = _as_float(data.get("x", data.get("world_x", data.get("posA"))))
    y = _as_float(data.get("y", data.get("world_y", data.get("posB"))))
    if x is None or y is None:
        return None
    return data


def probe_locidsub(
    host: str,
    port: int,
    loc_id: int,
    freq: float,
    wait_s: float,
) -> tuple[str, dict[str, Any] | str | None]:
    req = {
        "reqORinfo": "info",
        "type": "locIDsub",
        "locID": int(loc_id),
        "freq": float(freq),
    }
    try:
        with socket.create_connection((host, int(port)), timeout=3.0) as sock:
            sock.settimeout(0.25)
            sock.sendall(json.dumps(req).encode("utf-8"))
            deadline = time.monotonic() + max(wait_s, 0.1)
            buffer = ""
            last_message: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                objects, buffer = _extract_json_objects(buffer)
                for obj in objects:
                    last_message = obj
                    if obj.get("error") is not None:
                        return "error", obj
                    location = _location_like(obj)
                    if location is not None:
                        return "active", location
            return "silent", last_message
    except OSError as exc:
        return "connect_error", str(exc)


def _format_location(data: dict[str, Any], scale: float) -> str:
    x = _as_float(data.get("x", data.get("world_x", data.get("posA"))))
    y = _as_float(data.get("y", data.get("world_y", data.get("posB"))))
    z = _as_float(data.get("z"))
    parts = [f"raw=({x}, {y})"]
    if x is not None and y is not None:
        parts.append(f"m=({x * scale:.3f}, {y * scale:.3f})")
    if z is not None:
        parts.append(f"z_raw={z}")
        parts.append(f"z_m={z * scale:.3f}")
    if data.get("quality") is not None:
        parts.append(f"quality={data.get('quality')}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe active locIDsub IDs on a BFMC TrafficCommunicationServer."
    )
    parser.add_argument(
        "--ids",
        default="1-22",
        help="IDs to probe. Supports ranges and commas, e.g. 1-22 or 1,5,7-10.",
    )
    parser.add_argument(
        "--host",
        default="auto",
        help="TrafficCommunicationServer host. Default: auto discovery.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(_config_value("TRAFFIC_COMM_PORT", 5000)),
        help="TrafficCommunicationServer TCP port when --host is not auto.",
    )
    parser.add_argument(
        "--discovery-port",
        type=int,
        default=int(_config_value("TRAFFIC_COMM_DISCOVERY_PORT", 9000)),
        help="UDP port used by TrafficCommunicationServer broadcast.",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=float(_config_value("TRAFFIC_COMM_DISCOVERY_TIMEOUT_S", 5.0)),
        help="Seconds to wait for TrafficCommunicationServer autodiscovery.",
    )
    parser.add_argument(
        "--public-key",
        default=str(_config_value("TRAFFIC_COMM_PUBLIC_KEY_PATH", "auto")),
        help="Public key path for signed discovery, or auto.",
    )
    parser.add_argument(
        "--allow-unsigned-discovery",
        action="store_true",
        help="Accept unsigned broadcast packets. Use only for local debugging.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=float(_config_value("TRAFFIC_COMM_LOCSYS_SUB_FREQ", 0.25)),
        help="locIDsub freq value sent to the server.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1.5,
        help="Seconds to wait for a location message per ID.",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=float(_config_value("TRAFFIC_COMM_LOCSYS_SUB_COORD_SCALE", 0.001)),
        help="Scale for displayed coordinates. BFMC locIDsub is usually millimeters.",
    )
    args = parser.parse_args()

    host_arg = str(args.host).strip()
    if host_arg.lower() in {"", "auto", "discover", "autodiscover", "autodiscovery"}:
        host, port = discover_traffic_server(
            discovery_port=args.discovery_port,
            timeout_s=args.discovery_timeout,
            public_key_path=args.public_key,
            allow_unsigned=bool(args.allow_unsigned_discovery),
        )
        print(f"TrafficCommunicationServer descubierto en {host}:{port}")
    else:
        host, port = host_arg, int(args.port)
        print(f"Usando TrafficCommunicationServer manual {host}:{port}")

    ids = _parse_ids(args.ids)
    if not ids:
        raise SystemExit("No IDs to probe.")

    active_ids: list[int] = []
    print(f"Probando locIDsub IDs: {', '.join(str(i) for i in ids)}")
    for loc_id in ids:
        status, payload = probe_locidsub(
            host=host,
            port=port,
            loc_id=loc_id,
            freq=float(args.freq),
            wait_s=float(args.wait),
        )
        if status == "active" and isinstance(payload, dict):
            active_ids.append(loc_id)
            print(
                f"[ACTIVE] id={loc_id:02d} "
                f"{_format_location(payload, float(args.coord_scale))}"
            )
        elif status == "error":
            print(f"[ERROR ] id={loc_id:02d} response={payload!r}")
        elif status == "connect_error":
            print(f"[CONN  ] id={loc_id:02d} error={payload}")
        else:
            if payload is None:
                print(f"[silent] id={loc_id:02d} sin location en {args.wait:.1f}s")
            else:
                print(f"[silent] id={loc_id:02d} ultimo_mensaje={payload!r}")

    if active_ids:
        print("IDs activos:", ", ".join(str(i) for i in active_ids))
    else:
        print("IDs activos: ninguno detectado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
