"""
Cliente HybridNets para Raspberry Pi.

Se conecta al AI Server remoto por WebSocket, envía frames de la cámara
y recibe resultados de detección de carril + ángulo de dirección.

Puede usarse:
  1. Como módulo importado desde threadLineFollowing.py
  2. Como script standalone para testing

Uso standalone:
  python client.py --server ws://192.168.1.100:8500/ws/steering --test-image test.jpg
"""

import asyncio
import json
import time
import threading
import queue
import cv2
import numpy as np

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[AIClient] WARN: websockets no instalado. Ejecuta: pip install websockets")


class HybridNetsClient:
    """
    Cliente asíncrono para comunicarse con el AI Server de HybridNets.
    
    Maneja la conexión WebSocket en un thread separado para no bloquear
    el loop principal de la Raspberry Pi.
    
    Uso:
        client = HybridNetsClient("ws://192.168.1.100:8500/ws/steering")
        client.start()
        
        # En el loop principal:
        result = client.send_frame(frame)  # Bloquea hasta recibir respuesta
        steering = result['s']             # Ángulo de dirección
        
        client.stop()
    """
    
    def __init__(self, server_url: str = "ws://192.168.1.100:8500/ws/steering",
                 jpeg_quality: int = 70,
                 timeout: float = 2.0,
                 reconnect_interval: float = 3.0,
                 mode: str = "steering"):
        """
        Args:
            server_url: URL del WebSocket del servidor.
                        Usar /ws/steering para solo dirección (rápido)
                        Usar /ws/inference para resultados completos
            jpeg_quality: Calidad JPEG para comprimir frames (1-100). 
                         Menor = más rápido pero menos calidad.
            timeout: Timeout en segundos para esperar respuesta del servidor.
                     Con GPU (GTX 1060) la inferencia toma <100ms. Default: 2s.
            reconnect_interval: Segundos entre intentos de reconexión.
            mode: "steering" o "full". Steering es más rápido.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("Instala websockets: pip install websockets")
        
        self.server_url = server_url
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        self.mode = mode
        
        # Colas thread-safe para comunicación
        self._frame_queue = queue.Queue(maxsize=1)  # Solo el frame más reciente
        self._result_queue = queue.Queue(maxsize=1)
        
        # Estado
        self._running = False
        self._connected = False
        self._thread = None
        self._loop = None
        self._ws = None
        
        # Estadísticas
        self.frames_sent = 0
        self.frames_received = 0
        self.avg_roundtrip_ms = 0
        self._roundtrip_times = []
        
        # Último resultado (para acceso no-bloqueante)
        self._last_result = None
        self._last_result_time = 0
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    @property
    def last_result(self):
        """Último resultado recibido (puede ser None)."""
        return self._last_result
    
    def start(self):
        """Iniciar el cliente en un thread separado."""
        if self._running:
            return
        
        # Limpiar estado viejo antes de arrancar
        self.flush()
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AIClient")
        self._thread.start()
        print(f"[AIClient] Iniciado. Conectando a {self.server_url}")
    
    def stop(self):
        """Detener el cliente y limpiar estado."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._connected = False
        # Limpiar colas y resultados viejos para que al re-arrancar no procese datos stale
        self.flush()
        print("[AIClient] Detenido")
    
    def flush(self):
        """Limpiar colas y resultados viejos. Llamar al cambiar de modo."""
        # Vaciar cola de frames pendientes
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        # Vaciar cola de resultados pendientes
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        # Limpiar ultimo resultado (evita devolver datos viejos)
        self._last_result = None
        self._last_result_time = 0
    
    def send_frame(self, frame: np.ndarray, block: bool = True) -> dict:
        """
        Enviar un frame al servidor y obtener resultados.
        
        Args:
            frame: Imagen BGR de OpenCV
            block: Si True, bloquea hasta recibir respuesta (o timeout)
                   Si False, devuelve el último resultado disponible
                   
        Returns:
            dict con resultados del servidor, o None si no hay respuesta.
            
            Para modo "steering" (/ws/steering):
              {'s': float, 'c': float, 'e': float, 't': float, 'f': int}
              donde s=steering, c=confidence, e=error_norm, t=time_ms, f=frame_id
              
            Para modo "full" (/ws/inference):
              {'steering': {...}, 'lane_points': [...], 'detections': [...], ...}
        """
        if not self._connected:
            return None
        
        # Comprimir frame a JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_params)
        jpeg_bytes = encoded.tobytes()
        
        # Poner en la cola (descartar frame anterior si hay)
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        self._frame_queue.put(jpeg_bytes)
        
        if not block:
            return self._last_result
        
        # Esperar respuesta
        try:
            result = self._result_queue.get(timeout=self.timeout)
            return result
        except queue.Empty:
            # Timeout: no devolver resultado viejo, devolver None
            return None
    
    def send_frame_async(self, frame: np.ndarray):
        """
        Enviar frame sin esperar respuesta (fire-and-forget).
        El resultado estará disponible en self.last_result.
        """
        return self.send_frame(frame, block=False)
    
    def _run_loop(self):
        """Thread principal del cliente con asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connection_loop())
    
    async def _connection_loop(self):
        """Loop de conexión con reconexión automática."""
        while self._running:
            try:
                await self._connect_and_process()
            except Exception as e:
                if self._running:
                    print(f"[AIClient] Conexión perdida: {e}")
                    print(f"[AIClient] Reconectando en {self.reconnect_interval}s...")
                    self._connected = False
                    await asyncio.sleep(self.reconnect_interval)
    
    async def _connect_and_process(self):
        """Conectar y procesar frames."""
        try:
            async with websockets.connect(
                self.server_url,
                max_size=16 * 1024 * 1024,
                ping_interval=30,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                self._ws = ws
                self._connected = True
                consecutive_timeouts = 0
                print(f"[AIClient] Conectado a {self.server_url} (timeout={self.timeout}s)")
                
                while self._running:
                    # Esperar un frame para enviar
                    try:
                        jpeg_bytes = self._frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    send_time = time.time()
                    
                    # Enviar frame
                    await ws.send(jpeg_bytes)
                    self.frames_sent += 1
                    
                    # Recibir respuesta (con manejo de timeout que NO mata la conexión)
                    try:
                        response_data = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                    except asyncio.TimeoutError:
                        consecutive_timeouts += 1
                        elapsed = (time.time() - send_time) * 1000
                        print(f"[AIClient] Timeout esperando respuesta ({elapsed:.0f}ms > {self.timeout}s). "
                              f"Consecutivos: {consecutive_timeouts}")
                        if consecutive_timeouts >= 3:
                            print(f"[AIClient] {consecutive_timeouts} timeouts consecutivos, reconectando...")
                            break
                        # No romper la conexión — descartar este frame y seguir
                        continue
                    
                    consecutive_timeouts = 0
                    roundtrip_ms = (time.time() - send_time) * 1000
                    
                    # Parsear respuesta
                    try:
                        if isinstance(response_data, bytes):
                            try:
                                import msgpack
                                result = msgpack.unpackb(response_data, raw=False)
                            except (ImportError, Exception):
                                result = json.loads(response_data.decode())
                        else:
                            result = json.loads(response_data)
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"[AIClient] Error parseando respuesta: {e}")
                        continue
                    
                    # Agregar roundtrip time
                    result['roundtrip_ms'] = round(roundtrip_ms, 1)
                    
                    # Actualizar estado
                    self._last_result = result
                    self._last_result_time = time.time()
                    self.frames_received += 1
                    
                    # Log de los primeros frames
                    if self.frames_received <= 3:
                        print(f"[AIClient] Frame {self.frames_received}: roundtrip={roundtrip_ms:.0f}ms, "
                              f"server_time={result.get('t', result.get('inference_time_ms', '?'))}ms")
                    
                    # Actualizar estadísticas
                    self._roundtrip_times.append(roundtrip_ms)
                    if len(self._roundtrip_times) > 100:
                        self._roundtrip_times.pop(0)
                    self.avg_roundtrip_ms = sum(self._roundtrip_times) / len(self._roundtrip_times)
                    
                    # Poner resultado en la cola
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._result_queue.put(result)
                    
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            raise
        except ConnectionRefusedError:
            self._connected = False
            raise
        except OSError as e:
            self._connected = False
            raise
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del cliente."""
        return {
            'connected': self._connected,
            'server_url': self.server_url,
            'frames_sent': self.frames_sent,
            'frames_received': self.frames_received,
            'avg_roundtrip_ms': round(self.avg_roundtrip_ms, 1),
            'jpeg_quality': self.jpeg_quality,
            'last_result_age_ms': round((time.time() - self._last_result_time) * 1000, 1) if self._last_result_time else None,
        }


# ============================================================
# Test standalone
# ============================================================

def test_with_image(server_url: str, image_path: str):
    """Probar el cliente con una imagen estática."""
    print(f"Probando con imagen: {image_path}")
    print(f"Servidor: {server_url}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: No se pudo leer la imagen: {image_path}")
        return
    
    client = HybridNetsClient(server_url=server_url)
    client.start()
    
    # Esperar conexión
    for _ in range(30):
        if client.connected:
            break
        time.sleep(0.1)
    
    if not client.connected:
        print("ERROR: No se pudo conectar al servidor")
        client.stop()
        return
    
    print("Conectado! Enviando frame...")
    
    # Enviar varias veces para benchmark
    for i in range(10):
        result = client.send_frame(frame, block=True)
        if result:
            if 's' in result:
                # Modo steering
                print(f"  Frame {i+1}: steering={result.get('s')}, "
                      f"confidence={result.get('c')}, "
                      f"time={result.get('t')}ms, "
                      f"roundtrip={result.get('roundtrip_ms')}ms")
            else:
                # Modo full
                steering = result.get('steering', {})
                print(f"  Frame {i+1}: steering={steering.get('steering_angle')}, "
                      f"confidence={steering.get('confidence')}, "
                      f"inference={result.get('inference_time_ms')}ms, "
                      f"roundtrip={result.get('roundtrip_ms')}ms")
        else:
            print(f"  Frame {i+1}: Sin respuesta")
    
    print(f"\nEstadísticas: {client.get_stats()}")
    client.stop()


def test_with_camera(server_url: str, camera_index: int = 0):
    """Probar el cliente con la cámara en vivo."""
    print(f"Probando con cámara: {camera_index}")
    print(f"Servidor: {server_url}")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        return
    
    client = HybridNetsClient(server_url=server_url)
    client.start()
    
    # Esperar conexión
    for _ in range(30):
        if client.connected:
            break
        time.sleep(0.1)
    
    if not client.connected:
        print("ERROR: No se pudo conectar al servidor")
        client.stop()
        cap.release()
        return
    
    print("Conectado! Presiona 'q' para salir")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = client.send_frame(frame, block=True)
            
            if result:
                # Dibujar resultados en el frame
                if 's' in result:
                    steering = result.get('s')
                    confidence = result.get('c', 0)
                    rt = result.get('roundtrip_ms', 0)
                else:
                    steering_info = result.get('steering', {})
                    steering = steering_info.get('steering_angle')
                    confidence = steering_info.get('confidence', 0)
                    rt = result.get('roundtrip_ms', 0)
                
                # Overlay de info
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Steering: {steering}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"Roundtrip: {rt}ms", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Indicador visual de dirección
                center_x = w // 2
                if steering is not None:
                    offset = int(steering * 5)
                    cv2.arrowedLine(frame, (center_x, h - 30), 
                                   (center_x + offset, h - 80),
                                   (0, 255, 0), 3)
            
            cv2.imshow("HybridNets Client", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HybridNets AI Client")
    parser.add_argument("--server", type=str, 
                       default="ws://192.168.1.100:8500/ws/steering",
                       help="URL del WebSocket del servidor")
    parser.add_argument("--test-image", type=str, default=None,
                       help="Ruta a imagen para test")
    parser.add_argument("--camera", type=int, default=None,
                       help="Índice de cámara para test en vivo")
    parser.add_argument("--quality", type=int, default=70,
                       help="Calidad JPEG (1-100)")
    
    args = parser.parse_args()
    
    if args.test_image:
        test_with_image(args.server, args.test_image)
    elif args.camera is not None:
        test_with_camera(args.server, args.camera)
    else:
        print("Uso:")
        print(f"  python client.py --server {args.server} --test-image foto.jpg")
        print(f"  python client.py --server {args.server} --camera 0")
