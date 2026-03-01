"""
AI Server - Servidor de inferencia remota para detección de carriles.

Soporta múltiples motores de inferencia:
  - HybridNets: Segmentación + detección (requiere PyTorch + GPU)
  - Supercombo: Modelo de openpilot (requiere ONNX Runtime)
  - YOLO Lane Seg: Segmentación de carril izquierdo/derecho (requiere Ultralytics)

Seleccionar motor en config.py:
  ENGINE_TYPE = "hybridnets" | "supercombo" | "yolo_lane_seg"

La Raspberry Pi envía frames por WebSocket y este servidor responde
con los resultados de detección de carril y ángulo de dirección.

Protocolo WebSocket:
  - Cliente envía: bytes JPEG del frame
  - Servidor responde: msgpack/JSON con resultados

Endpoints HTTP:
  - GET /              -> Estado del servidor
  - GET /status        -> Info detallada del modelo y GPU
  - WS  /ws/inference  -> WebSocket para inferencia en tiempo real

Uso:
  python server.py
  # o
  uvicorn server:app --host 0.0.0.0 --port 8500
"""

import asyncio
import base64
import json
import time
import io
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse

import config

# ============================================================
# Variables globales de motores de inferencia
# ============================================================
engine = None
sign_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown del servidor."""
    global engine, sign_engine
    engine_type = getattr(config, 'ENGINE_TYPE', 'hybridnets')
    sign_enabled = getattr(config, 'SIGN_DETECTION_ENABLED', False)
    
    print("=" * 60)
    print(f"  AI Server - Iniciando ({engine_type})")
    if sign_enabled:
        print(f"  + Sign Detection (YOLOv8)")
    print("=" * 60)
    
    # Cargar motor de inferencia de carriles según configuración
    if engine_type == "supercombo":
        from supercombo_engine import SupercomboEngine
        engine = SupercomboEngine()
    elif engine_type == "yolo_lane_seg":
        from yolo_lane_seg_engine import YOLOLaneSegEngine
        engine = YOLOLaneSegEngine()
    else:
        from inference import HybridNetsEngine
        engine = HybridNetsEngine()
    
    # Cargar motor de detección de señales (opcional)
    if sign_enabled:
        try:
            from sign_detection_engine import SignDetectionEngine
            sign_engine = SignDetectionEngine()
        except Exception as e:
            print(f"[Server] WARNING: Sign detection disabled: {e}")
            sign_engine = None
    
    show_viz = getattr(config, 'SHOW_VISUALIZATION', False)
    print("=" * 60)
    print(f"  Servidor listo en {config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"  Motor carriles: {engine_type}")
    print(f"  Motor señales:  {'OK' if sign_engine else 'disabled'}")
    print(f"  WS carriles:  ws://<ip>:{config.SERVER_PORT}/ws/steering")
    if sign_engine:
        print(f"  WS señales:   ws://<ip>:{config.SERVER_PORT}/ws/signs")
    if show_viz and (sign_engine or (engine is not None and hasattr(engine, "get_vis_jpeg"))):
        print(f"  Visualización: http://localhost:{config.SERVER_PORT}/viz")
    print("=" * 60)
    
    yield
    
    # Cleanup
    print("[Server] Apagando servidor...")
    if engine is not None and hasattr(engine, 'shutdown'):
        engine.shutdown()
    if sign_engine is not None and hasattr(sign_engine, 'shutdown'):
        sign_engine.shutdown()
    cv2.destroyAllWindows()
    engine = None
    sign_engine = None


app = FastAPI(
    title="AI Server",
    description="Servidor de inferencia remota para deteccion de carriles (HybridNets/Supercombo/YOLO Lane Seg)",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# Endpoints HTTP
# ============================================================

@app.get("/")
async def root():
    """Estado básico del servidor."""
    engine_type = getattr(config, 'ENGINE_TYPE', 'hybridnets')
    return {
        "status": "running",
        "model": engine_type,
        "port": config.SERVER_PORT,
        "websocket_endpoint": f"ws://0.0.0.0:{config.SERVER_PORT}/ws/steering",
    }


@app.get("/status")
async def status():
    """Estado detallado del servidor y modelo."""
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Modelo no cargado aún"}
        )
    server_status = engine.get_status()
    server_status["sign_detection_enabled"] = sign_engine is not None
    if sign_engine is not None:
        server_status["sign_detection"] = sign_engine.get_status()
    return server_status


# ============================================================
# WebSocket para inferencia en tiempo real
# ============================================================

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    """
    WebSocket endpoint para inferencia en tiempo real.
    
    Protocolo:
      1. Cliente envía frame como bytes JPEG
      2. Servidor procesa con HybridNets
      3. Servidor responde con resultados
      
    Formato de respuesta (msgpack o JSON):
    {
        "steering": {
            "steering_angle": float,    # -25 a 25 grados
            "lane_center_x": float,     # posición X del centro
            "confidence": float,        # 0-1
            "error_normalized": float,  # -1 a 1
        },
        "lane_points": [               # Puntos de las líneas
            [(x1,y1), (x2,y2), ...],   # Línea izquierda
            [(x1,y1), (x2,y2), ...],   # Línea derecha
        ],
        "detections": [...],           # Detecciones de objetos
        "inference_time_ms": float,
        "frame_id": int,
        "lane_mask_b64": str,          # Máscara PNG en base64
        "road_mask_b64": str,          # Máscara PNG en base64
    }
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[Server] Cliente conectado: {client_host}")
    
    if engine is None:
        await websocket.send_json({"error": "Modelo no cargado"})
        await websocket.close()
        return
    
    frames_processed = 0
    total_time = 0
    
    try:
        while True:
            # Recibir frame como bytes
            data = await websocket.receive_bytes()
            
            recv_time = time.time()
            
            # Decodificar JPEG
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Frame inválido"})
                continue
            
            # Ejecutar inferencia en thread pool para no bloquear el event loop
            # (evita que ping/pong de WebSocket fallen y causen desconexiones)
            infer_start = time.time()
            result = await asyncio.to_thread(engine.infer, frame)
            infer_ms = (time.time() - infer_start) * 1000
            
            # Log de tiempo para los primeros frames (para diagnóstico)
            if frames_processed < 5:
                print(f"[Server] Frame {frames_processed + 1}: "
                      f"inferencia={infer_ms:.0f}ms, "
                      f"res={frame.shape[1]}x{frame.shape[0]}")
            
            # Preparar respuesta
            response = {
                'steering': result['steering'],
                'lane_points': result['lane_points'],
                'detections': result['detections'],
                'inference_time_ms': result['inference_time_ms'],
                'frame_id': result['frame_id'],
                'input_size': result['input_size'],
            }
            
            # Agregar máscaras en base64 (si el cliente las necesita)
            if result['lane_mask']:
                response['lane_mask_b64'] = base64.b64encode(result['lane_mask']).decode('ascii')
            if result['road_mask']:
                response['road_mask_b64'] = base64.b64encode(result['road_mask']).decode('ascii')
            
            # Agregar latencia total
            total_process_time = (time.time() - recv_time) * 1000
            response['total_server_time_ms'] = round(total_process_time, 1)
            
            # Estadísticas
            frames_processed += 1
            total_time += total_process_time
            
            if frames_processed % 100 == 0:
                avg_time = total_time / frames_processed
                print(f"[Server] Cliente {client_host}: {frames_processed} frames, "
                      f"avg {avg_time:.1f}ms/frame ({1000/avg_time:.1f} FPS)")
            
            # Enviar respuesta
            if config.RESPONSE_FORMAT == "msgpack":
                try:
                    import msgpack
                    packed = msgpack.packb(response, use_bin_type=True)
                    await websocket.send_bytes(packed)
                except ImportError:
                    # Fallback a JSON si msgpack no está instalado
                    await websocket.send_json(response)
            else:
                await websocket.send_json(response)
    
    except WebSocketDisconnect:
        print(f"[Server] Cliente desconectado: {client_host} "
              f"({frames_processed} frames procesados)")
    except Exception as e:
        print(f"[Server] Error con cliente {client_host}: {e}")
        try:
            await websocket.close()
        except:
            pass


# ============================================================
# WebSocket ligero (solo steering, sin máscaras)
# ============================================================

@app.websocket("/ws/steering")
async def websocket_steering_only(websocket: WebSocket):
    """
    WebSocket optimizado que solo devuelve el ángulo de dirección.
    Mucho menor latencia y ancho de banda.
    
    Respuesta: JSON mínimo
    {
        "s": float,   # steering_angle
        "c": float,   # confidence
        "t": float,   # inference_time_ms
        "f": int,      # frame_id
    }
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[Server] Cliente steering conectado: {client_host}")
    
    if engine is None:
        await websocket.send_json({"error": "Modelo no cargado"})
        await websocket.close()
        return
    
    frames_processed = 0
    total_infer_ms = 0
    total_roundtrip_ms = 0
    
    try:
        while True:
            t_recv_start = time.time()
            data = await websocket.receive_bytes()
            t_recv = time.time()
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            t_decode = time.time()
            
            if frame is None:
                await websocket.send_text('{"error":"bad frame"}')
                continue
            
            # Usar infer_steering_only para evitar trabajo innecesario
            # (no comprime mascaras, no post-procesa detecciones)
            t_infer_start = time.time()
            result = await asyncio.to_thread(engine.infer_steering_only, frame)
            t_infer_end = time.time()
            
            # Respuesta mínima
            steering = result['steering']
            response = {
                's': steering['steering_angle'],
                'c': steering['confidence'],
                'e': steering['error_normalized'],
                't': result['inference_time_ms'],
                'f': result['frame_id'],
            }
            
            # Enviar como texto JSON compacto (más rápido que msgpack para payloads pequeños)
            response_text = json.dumps(response, separators=(',', ':'))
            await websocket.send_text(response_text)
            t_sent = time.time()
            
            # === TIMING LOGS ===
            frames_processed += 1
            decode_ms = (t_decode - t_recv) * 1000
            infer_ms = (t_infer_end - t_infer_start) * 1000
            send_ms = (t_sent - t_infer_end) * 1000
            total_ms = (t_sent - t_recv) * 1000
            total_infer_ms += infer_ms
            total_roundtrip_ms += total_ms
            
            # Log detallado para los primeros 10 frames
            if frames_processed <= 10:
                print(f"[Steering] Frame {frames_processed}: "
                      f"jpeg_decode={decode_ms:.1f}ms | "
                      f"infer={infer_ms:.1f}ms (engine={result['inference_time_ms']}ms) | "
                      f"send={send_ms:.1f}ms | "
                      f"TOTAL={total_ms:.1f}ms | "
                      f"frame={frame.shape[1]}x{frame.shape[0]} ({len(data)} bytes)")
            # Log resumido cada 50 frames
            elif frames_processed % 50 == 0:
                avg_infer = total_infer_ms / frames_processed
                avg_total = total_roundtrip_ms / frames_processed
                fps = 1000 / avg_total if avg_total > 0 else 0
                print(f"[Steering] {frames_processed} frames | "
                      f"avg_infer={avg_infer:.1f}ms | "
                      f"avg_total={avg_total:.1f}ms | "
                      f"~{fps:.1f} FPS")
    
    except WebSocketDisconnect:
        if frames_processed > 0:
            avg_infer = total_infer_ms / frames_processed
            avg_total = total_roundtrip_ms / frames_processed
            print(f"[Steering] Cliente desconectado: {client_host} | "
                  f"{frames_processed} frames | avg_infer={avg_infer:.1f}ms | avg_total={avg_total:.1f}ms")
        else:
            print(f"[Server] Cliente steering desconectado: {client_host}")
    except Exception as e:
        print(f"[Server] Error con cliente steering {client_host}: {e}")


# ============================================================
# WebSocket para detección de señales de tráfico
# ============================================================

@app.websocket("/ws/signs")
async def websocket_signs(websocket: WebSocket):
    """
    WebSocket para detección de señales de tráfico.
    
    Protocolo:
      - Cliente envía: bytes JPEG del frame
      - Servidor responde: JSON con detecciones
    
    Respuesta:
    {
        "d": [                        # detections (lista)
            {"s": "stop", "c": 0.95, "b": [y1,x1,y2,x2], "k": "sign", "m": "signs_640"},
            ...
        ],
        "t": 12.3,                    # inference_time_ms
        "f": 42                       # frame_id
    }
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[Signs] Cliente conectado: {client_host}")
    
    if sign_engine is None:
        await websocket.send_json({"error": "Sign detection no disponible"})
        await websocket.close()
        return
    
    frames_processed = 0
    total_infer_ms = 0
    total_roundtrip_ms = 0
    
    consecutive_errors = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            t_recv = time.time()  # Start timing AFTER receiving data
            
            try:
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_text('{"error":"decode failed"}')
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    print(f"[Signs] Demasiados errores de decode, cerrando {client_host}")
                    break
                continue
            
            if frame is None:
                await websocket.send_text('{"error":"bad frame"}')
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    break
                continue
            
            try:
                # Run sign detection in thread pool
                t_infer_start = time.time()
                result = await asyncio.to_thread(sign_engine.infer, frame)
                t_infer_end = time.time()
            except Exception as infer_err:
                print(f"[Signs] Inference error: {infer_err}")
                await websocket.send_text('{"d":[],"t":0,"f":0}')
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print(f"[Signs] Demasiados errores de inferencia, cerrando {client_host}")
                    break
                continue
            
            consecutive_errors = 0  # Reset on success
            
            # Compact response (minimize bandwidth)
            compact_dets = []
            for det in result["detections"]:
                compact_dets.append({
                    "s": det["sign"],
                    "c": det["confidence"],
                    "b": det["box"],
                    "k": det.get("kind", "sign"),
                    "m": det.get("model", "default"),
                })
            
            response = {
                "d": compact_dets,
                "t": result["inference_time_ms"],
                "f": result["frame_id"],
            }
            
            response_text = json.dumps(response, separators=(",", ":"))
            await websocket.send_text(response_text)
            t_sent = time.time()
            
            # Timing
            frames_processed += 1
            infer_ms = (t_infer_end - t_infer_start) * 1000
            total_ms = (t_sent - t_recv) * 1000
            total_infer_ms += infer_ms
            total_roundtrip_ms += total_ms
            
            if frames_processed <= 5:
                print(f"[Signs] Frame {frames_processed}: "
                      f"infer={infer_ms:.1f}ms | total={total_ms:.1f}ms | "
                      f"dets={len(compact_dets)} | "
                      f"frame={frame.shape[1]}x{frame.shape[0]} ({len(data)} bytes)")
            elif frames_processed % 50 == 0:
                avg_infer = total_infer_ms / frames_processed
                avg_total = total_roundtrip_ms / frames_processed
                fps = 1000 / avg_total if avg_total > 0 else 0
                print(f"[Signs] {frames_processed} frames | "
                      f"avg_infer={avg_infer:.1f}ms | avg_total={avg_total:.1f}ms | "
                      f"~{fps:.1f} FPS")
    
    except WebSocketDisconnect:
        if frames_processed > 0:
            avg = total_infer_ms / frames_processed
            print(f"[Signs] Cliente desconectado: {client_host} | "
                  f"{frames_processed} frames | avg_infer={avg:.1f}ms")
        else:
            print(f"[Signs] Cliente desconectado: {client_host}")
    except Exception as e:
        print(f"[Signs] Error con cliente {client_host}: {e}")


# ============================================================
# Visualización MJPEG (bounding boxes en el navegador)
# ============================================================

@app.get("/viz/lanes")
async def viz_lanes_stream():
    """MJPEG stream: video en vivo con overlay de carriles detectados."""
    if (
        engine is None
        or not hasattr(engine, "get_vis_jpeg")
        or not getattr(engine, "show_visualization", False)
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Visualizacion de carriles no disponible"},
        )

    async def generate():
        while True:
            jpeg = engine.get_vis_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/viz/signs")
async def viz_signs_stream():
    """MJPEG stream: video en vivo con bounding boxes de señales detectadas.

    Abre esta URL en cualquier navegador (funciona en macOS, Linux, Windows).
    El stream se activa cuando un cliente WebSocket envía frames a /ws/signs.
    """
    if sign_engine is None or not getattr(sign_engine, "show_visualization", False):
        return JSONResponse(
            status_code=503,
            content={"error": "Sign detection no disponible"},
        )

    async def generate():
        while True:
            jpeg = sign_engine.get_vis_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/viz", response_class=HTMLResponse)
async def viz_page():
    """Página HTML con visualización en vivo de carriles y señales."""
    port = config.SERVER_PORT
    lane_available = (
        engine is not None
        and hasattr(engine, "get_vis_jpeg")
        and getattr(engine, "show_visualization", False)
    )
    sign_available = sign_engine is not None and getattr(sign_engine, "show_visualization", False)
    lane_section = ""
    if lane_available:
        lane_section = """
    <section class="panel">
      <div class="panel-header">
        <h2>Carriles</h2>
        <span class="badge">/viz/lanes</span>
      </div>
      <div class="panel-body">
        <div class="waiting" id="waiting-lanes">
          <div class="spinner"></div>
          <div>Esperando frames de /ws/steering...</div>
        </div>
        <img class="stream" src="/viz/lanes" alt="Lane Stream"
             style="display:none"
             onload="showStream('lanes', this)">
      </div>
    </section>"""
    sign_section = ""
    if sign_available:
        sign_section = """
    <section class="panel">
      <div class="panel-header">
        <h2>Señales</h2>
        <span class="badge">/viz/signs</span>
      </div>
      <div class="panel-body">
        <div class="waiting" id="waiting-signs">
          <div class="spinner"></div>
          <div>Esperando frames de /ws/signs...</div>
        </div>
        <img class="stream" src="/viz/signs" alt="Sign Stream"
             style="display:none"
             onload="showStream('signs', this)">
      </div>
    </section>"""
    if not lane_section and not sign_section:
        lane_section = """
    <section class="panel">
      <div class="panel-body">
        <div class="waiting" style="display:block">
          <div class="spinner"></div>
          <div>No hay streams de visualizacion disponibles.</div>
        </div>
      </div>
    </section>"""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>AI Server Viz</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      background: #0f0f1a;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }}
    header {{
      width: 100%;
      padding: 14px 24px;
      background: #1a1a2e;
      border-bottom: 1px solid #2a2a4a;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    header h1 {{
      font-size: 1.1em;
      font-weight: 600;
      color: #00e5ff;
    }}
    main {{
      width: 100%;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
      padding: 16px;
    }}
    .panel {{
      background: #151528;
      border: 1px solid #2a2a4a;
      border-radius: 10px;
      overflow: hidden;
      min-height: 280px;
    }}
    .panel-header {{
      padding: 12px 14px;
      border-bottom: 1px solid #2a2a4a;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    .panel-header h2 {{
      font-size: 0.95em;
      font-weight: 600;
      color: #9be7ff;
    }}
    .panel-body {{
      padding: 12px;
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }}
    header .badge {{
      font-size: 0.75em;
      padding: 2px 8px;
      border-radius: 4px;
      background: #00e5ff22;
      color: #00e5ff;
      border: 1px solid #00e5ff44;
    }}
    .stream {{
      width: 100%;
      max-height: 72vh;
      object-fit: contain;
      border: 2px solid #2a2a4a;
      border-radius: 8px;
      background: #1a1a2e;
    }}
    footer {{
      padding: 10px;
      font-size: 0.8em;
      color: #555;
    }}
    .waiting {{
      color: #888;
      font-size: 1.1em;
      text-align: center;
      padding: 24px;
    }}
    .waiting .spinner {{
      display: inline-block;
      width: 24px;
      height: 24px;
      border: 3px solid #333;
      border-top: 3px solid #00e5ff;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin-bottom: 12px;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
  <script>
    function showStream(name, img) {{
      const waiting = document.getElementById('waiting-' + name);
      if (waiting) waiting.style.display = 'none';
      const status = document.getElementById('status');
      status.textContent = 'Live';
      status.style.background = '#00c85322';
      status.style.color = '#00c853';
      status.style.borderColor = '#00c85344';
      if (img) img.style.display = 'block';
    }}
  </script>
</head>
<body>
  <header>
    <h1>AI Server Visualization</h1>
    <span class="badge">AI Server :{port}</span>
    <span class="badge" id="status">Connecting...</span>
  </header>
  <main>
    {lane_section}
    {sign_section}
  </main>
  <footer>MJPEG stream server-side para carriles y señales</footer>
</body>
</html>"""


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info",
        ws_max_size=16 * 1024 * 1024,  # 16MB max WebSocket message
    )
