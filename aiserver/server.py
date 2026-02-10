"""
AI Server para HybridNets - Servidor de inferencia remota.

La Raspberry Pi envía frames por WebSocket y este servidor responde
con los resultados de segmentación de carril y ángulo de dirección.

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
from fastapi.responses import JSONResponse

import config

# ============================================================
# Variable global del motor de inferencia
# ============================================================
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown del servidor."""
    global engine
    print("=" * 60)
    print("  AI Server para HybridNets - Iniciando")
    print("=" * 60)
    
    # Cargar modelo
    from inference import HybridNetsEngine
    engine = HybridNetsEngine()
    
    print("=" * 60)
    print(f"  Servidor listo en {config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"  WebSocket: ws://<ip>:{config.SERVER_PORT}/ws/inference")
    print("=" * 60)
    
    yield
    
    # Cleanup
    print("[Server] Apagando servidor...")
    engine = None


app = FastAPI(
    title="HybridNets AI Server",
    description="Servidor de inferencia remota para detección de carriles con HybridNets",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Endpoints HTTP
# ============================================================

@app.get("/")
async def root():
    """Estado básico del servidor."""
    return {
        "status": "running",
        "model": "HybridNets",
        "port": config.SERVER_PORT,
        "websocket_endpoint": f"ws://0.0.0.0:{config.SERVER_PORT}/ws/inference",
    }


@app.get("/status")
async def status():
    """Estado detallado del servidor y modelo."""
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Modelo no cargado aún"}
        )
    return engine.get_status()


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
            
            # Ejecutar inferencia
            infer_start = time.time()
            result = engine.infer(frame)
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
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_text('{"error":"bad frame"}')
                continue
            
            result = engine.infer(frame)
            
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
            await websocket.send_text(json.dumps(response, separators=(',', ':')))
    
    except WebSocketDisconnect:
        print(f"[Server] Cliente steering desconectado: {client_host}")
    except Exception as e:
        print(f"[Server] Error con cliente steering {client_host}: {e}")


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
