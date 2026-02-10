"""
Motor de inferencia HybridNets.
Carga el modelo, procesa frames y devuelve resultados de:
  - Detección de objetos (autos, etc.)
  - Segmentación de área manejable (drivable area)
  - Detección de líneas de carril (lane lines)
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import config

# ============================================================
# Agregar el repo HybridNets al path (se clona en setup)
# ============================================================
HYBRIDNETS_DIR = os.path.join(os.path.dirname(__file__), "HybridNets")
if os.path.isdir(HYBRIDNETS_DIR):
    sys.path.insert(0, HYBRIDNETS_DIR)


class HybridNetsEngine:
    """
    Motor de inferencia para HybridNets.
    
    Carga el modelo pre-entrenado y realiza inferencia sobre frames individuales.
    Devuelve:
      - Máscara de segmentación (road + lane)
      - Puntos de líneas de carril
      - Detecciones de objetos (bounding boxes)
      - Ángulo de dirección sugerido
    """
    
    def __init__(self,
                 weights_path: str = None,
                 device: str = None,
                 input_width: int = None,
                 input_height: int = None,
                 use_half: bool = None):
        
        self.weights_path = weights_path or config.WEIGHTS_PATH
        self.device_str = device or config.DEVICE
        self.input_width = input_width or config.INPUT_WIDTH
        self.input_height = input_height or config.INPUT_HEIGHT
        self.use_half = use_half if use_half is not None else config.USE_HALF
        
        # Normalización ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Estado de seguimiento
        self.previous_steering = 0.0
        self.frame_count = 0
        
        # Modelo
        self.model = None
        self.device = None
        self.anchors = None
        
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo HybridNets."""
        print(f"[HybridNets] Cargando modelo desde: {self.weights_path}")
        print(f"[HybridNets] Dispositivo: {self.device_str}")
        print(f"[HybridNets] Resolución: {self.input_width}x{self.input_height}")
        
        # Seleccionar dispositivo
        if self.device_str.startswith("cuda") and not torch.cuda.is_available():
            print("[HybridNets] WARN: CUDA no disponible, usando CPU")
            self.device_str = "cpu"
            self.use_half = False
        
        if self.device_str == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("[HybridNets] WARN: MPS no disponible, usando CPU")
            self.device_str = "cpu"
            self.use_half = False
        
        if self.device_str == "cpu":
            self.use_half = False
        
        self.device = torch.device(self.device_str)
        
        # Intentar cargar el modelo
        try:
            self._load_hybridnets_model()
            print(f"[HybridNets] Modelo cargado exitosamente en {self.device_str}")
            if self.use_half:
                print("[HybridNets] Usando FP16 (half precision)")
        except Exception as e:
            print(f"[HybridNets] ERROR al cargar modelo: {e}")
            print("[HybridNets] Ejecuta: python setup_server.py para descargar el modelo")
            raise
    
    def _load_hybridnets_model(self):
        """Cargar modelo HybridNets desde torch.hub o desde archivo local."""
        
        # Verificar si existe el peso local
        weights_full_path = os.path.join(os.path.dirname(__file__), self.weights_path)
        
        if os.path.exists(weights_full_path):
            print(f"[HybridNets] Cargando peso local: {weights_full_path}")
            # Cargar desde torch.hub con peso local
            self.model = torch.hub.load(
                HYBRIDNETS_DIR if os.path.isdir(HYBRIDNETS_DIR) else 'datvuthanh/HybridNets',
                'hybridnets',
                pretrained=False,
                source='local' if os.path.isdir(HYBRIDNETS_DIR) else 'github',
            )
            # Cargar los pesos
            checkpoint = torch.load(weights_full_path, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("[HybridNets] Descargando modelo pre-entrenado desde GitHub...")
            self.model = torch.hub.load(
                'datvuthanh/HybridNets',
                'hybridnets',
                pretrained=True,
            )
        
        self.model.to(self.device)
        
        if self.use_half:
            self.model.half()
        
        self.model.eval()
        
        # Warm-up: correr una inferencia dummy y medir tiempo
        print("[HybridNets] Warm-up...")
        dummy = torch.zeros(1, 3, self.input_height, self.input_width).to(self.device)
        if self.use_half:
            dummy = dummy.half()
        
        warmup_start = time.time()
        with torch.no_grad():
            _ = self.model(dummy)
        warmup_ms = (time.time() - warmup_start) * 1000
        
        # Log de diagnostico
        print(f"[HybridNets] Warm-up completado en {warmup_ms:.0f}ms")
        print(f"[HybridNets] === DIAGNOSTICO ===")
        print(f"[HybridNets]   Device: {self.device} (CUDA available: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"[HybridNets]   GPU: {torch.cuda.get_device_name(0)}")
            mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**2
            print(f"[HybridNets]   VRAM: {mem_alloc:.0f}MB / {mem_total:.0f}MB")
        print(f"[HybridNets]   FP16: {self.use_half}")
        print(f"[HybridNets]   Input: {self.input_width}x{self.input_height}")
        print(f"[HybridNets]   Warmup forward: {warmup_ms:.0f}ms")
        print(f"[HybridNets] ==================")
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocesar frame para inferencia.
        
        Args:
            frame: Imagen BGR de OpenCV (HxWx3)
            
        Returns:
            Tensor normalizado (1x3xHxW)
        """
        # Redimensionar
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar [0,255] -> [0,1] -> ImageNet
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        
        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        if self.use_half:
            tensor = tensor.half()
        
        return tensor
    
    def _resolve_seg_tensor(self, seg_output):
        """
        Resuelve la salida de segmentación a un único torch.Tensor.
        HybridNets puede devolver la segmentación como:
          - torch.Tensor directamente
          - Tupla/lista de tensores (ej: (road_seg, lane_seg))
          - Tupla anidada
        """
        # Si ya es tensor, listo
        if isinstance(seg_output, torch.Tensor):
            return seg_output

        # Si es tupla o lista, buscar el tensor de segmentación dentro
        if isinstance(seg_output, (tuple, list)):
            if self.frame_count < 3:
                print(f"[HybridNets] seg_output es {type(seg_output).__name__} "
                      f"con {len(seg_output)} elementos")
                for i, item in enumerate(seg_output):
                    print(f"  [{i}] tipo={type(item).__name__}"
                          f"{f', shape={item.shape}' if isinstance(item, (torch.Tensor, np.ndarray)) else ''}")

            # Filtrar solo los tensores/arrays
            tensors = [t for t in seg_output if isinstance(t, (torch.Tensor, np.ndarray))]
            if not tensors:
                # Buscar un nivel más profundo (tupla de tuplas)
                for item in seg_output:
                    if isinstance(item, (tuple, list)):
                        for sub in item:
                            if isinstance(sub, (torch.Tensor, np.ndarray)):
                                tensors.append(sub)

            if len(tensors) == 1:
                t = tensors[0]
                return torch.as_tensor(t) if isinstance(t, np.ndarray) else t

            if len(tensors) >= 2:
                # Múltiples mapas de segmentación — apilarlos en dim de canales
                # para que argmax pueda elegir la clase
                stacked = []
                for t in tensors:
                    t_tensor = torch.as_tensor(t) if isinstance(t, np.ndarray) else t
                    # Asegurarse de que cada tensor tenga forma [1, 1, H, W]
                    while t_tensor.dim() < 4:
                        t_tensor = t_tensor.unsqueeze(0)
                    stacked.append(t_tensor)
                # Concatenar en dim=1 (canales): [1, N, H, W]
                return torch.cat(stacked, dim=1)

        # Último recurso: intentar convertir a tensor
        if self.frame_count < 3:
            print(f"[HybridNets] WARN: tipo de seg_output no esperado: {type(seg_output)}")
        return torch.as_tensor(np.array(seg_output))

    def postprocess_segmentation(self, seg_output, original_size: tuple):
        """
        Post-procesar salida de segmentación.
        
        Args:
            seg_output: Tensor de salida del modelo (segmentación)
            original_size: (height, width) del frame original
            
        Returns:
            dict con:
              - 'road_mask': máscara binaria del área manejable (HxW, uint8)
              - 'lane_mask': máscara binaria de líneas de carril (HxW, uint8)
              - 'lane_points': lista de puntos de las líneas detectadas
        """
        orig_h, orig_w = original_size
        
        # Resolver a tensor único (maneja tuplas, listas, etc.)
        seg_tensor = self._resolve_seg_tensor(seg_output)
        seg = seg_tensor.detach() if isinstance(seg_tensor, torch.Tensor) else seg_tensor

        # La salida de segmentación tiene forma [batch, classes, H, W]
        # classes: 0=background, 1=road, 2=lane
        if isinstance(seg, torch.Tensor):
            # Aplicar softmax o argmax
            if seg.dim() >= 2 and seg.shape[1] > 1:
                seg_pred = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy()
            else:
                seg_pred = (seg.squeeze(0).squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
        elif isinstance(seg, np.ndarray):
            seg_pred = seg
        else:
            print(f"[HybridNets] WARN: seg no es tensor ni ndarray: {type(seg)}")
            seg_pred = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # Redimensionar al tamaño original
        seg_resized = cv2.resize(seg_pred.astype(np.uint8), (orig_w, orig_h), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Extraer máscaras individuales
        road_mask = (seg_resized == 1).astype(np.uint8) * 255
        lane_mask = (seg_resized == 2).astype(np.uint8) * 255
        
        # Extraer puntos de las líneas de carril
        lane_points = self._extract_lane_points(lane_mask)
        
        return {
            'road_mask': road_mask,
            'lane_mask': lane_mask,
            'lane_points': lane_points,
            'seg_raw': seg_resized,
        }
    
    def _extract_lane_points(self, lane_mask: np.ndarray):
        """
        Extraer puntos de las líneas de carril desde la máscara de segmentación.
        
        Divide la máscara en filas horizontales y encuentra los centroides
        de cada cluster de píxeles blancos.
        
        Returns:
            Lista de líneas, donde cada línea es una lista de puntos (x, y)
        """
        h, w = lane_mask.shape[:2]
        
        # Muestrear en N filas
        n_samples = 20
        y_positions = np.linspace(int(h * 0.3), h - 1, n_samples).astype(int)
        
        all_points = []  # Lista de listas (una por fila)
        
        for y in y_positions:
            row = lane_mask[y, :]
            # Encontrar segmentos blancos
            white_pixels = np.where(row > 128)[0]
            
            if len(white_pixels) == 0:
                continue
            
            # Agrupar píxeles cercanos (separación > 30px = nueva línea)
            clusters = []
            current_cluster = [white_pixels[0]]
            
            for i in range(1, len(white_pixels)):
                if white_pixels[i] - white_pixels[i-1] > 30:
                    clusters.append(current_cluster)
                    current_cluster = [white_pixels[i]]
                else:
                    current_cluster.append(white_pixels[i])
            clusters.append(current_cluster)
            
            # Centroide de cada cluster
            for cluster in clusters:
                cx = int(np.mean(cluster))
                all_points.append((cx, int(y)))
        
        # Agrupar puntos en líneas usando proximidad en X
        if len(all_points) < 2:
            return []
        
        # Separar en izquierda y derecha del centro
        center_x = w // 2
        left_points = [(x, y) for x, y in all_points if x < center_x]
        right_points = [(x, y) for x, y in all_points if x >= center_x]
        
        lines = []
        if left_points:
            lines.append(sorted(left_points, key=lambda p: p[1]))
        if right_points:
            lines.append(sorted(right_points, key=lambda p: p[1]))
        
        return lines
    
    def compute_steering(self, lane_points: list, frame_width: int, frame_height: int) -> dict:
        """
        Calcular ángulo de dirección a partir de los puntos de carril detectados.
        
        Args:
            lane_points: Lista de líneas (cada una es lista de puntos)
            frame_width: Ancho del frame
            frame_height: Alto del frame
            
        Returns:
            dict con:
              - 'steering_angle': ángulo sugerido (-25 a 25 grados)
              - 'lane_center_x': posición X del centro del carril
              - 'confidence': confianza de la detección (0-1)
              - 'error_normalized': error normalizado (-1 a 1)
        """
        if not lane_points or len(lane_points) == 0:
            return {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            }
        
        frame_center = frame_width / 2
        y_target = int(frame_height * (1.0 - config.LOOKAHEAD_RATIO))
        
        # Encontrar el punto más cercano al y_target en cada línea
        lane_x_at_target = []
        for line in lane_points:
            if not line:
                continue
            # Encontrar punto más cercano al y_target
            closest = min(line, key=lambda p: abs(p[1] - y_target))
            lane_x_at_target.append(closest[0])
        
        if not lane_x_at_target:
            return {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            }
        
        # Calcular centro del carril
        if len(lane_x_at_target) >= 2:
            # Separar izquierda y derecha
            left_lanes = [x for x in lane_x_at_target if x < frame_center]
            right_lanes = [x for x in lane_x_at_target if x >= frame_center]
            
            if left_lanes and right_lanes:
                left_x = max(left_lanes)   # Línea más cercana al centro por la izquierda
                right_x = min(right_lanes)  # Línea más cercana al centro por la derecha
                lane_center = (left_x + right_x) / 2
                confidence = 1.0
            elif left_lanes:
                lane_center = max(left_lanes) + frame_width * config.LANE_WIDTH_ESTIMATE / 2
                confidence = 0.6
            else:
                lane_center = min(right_lanes) - frame_width * config.LANE_WIDTH_ESTIMATE / 2
                confidence = 0.6
        else:
            x = lane_x_at_target[0]
            if x < frame_center:
                lane_center = x + frame_width * config.LANE_WIDTH_ESTIMATE / 2
            else:
                lane_center = x - frame_width * config.LANE_WIDTH_ESTIMATE / 2
            confidence = 0.5
        
        # Calcular error normalizado
        error = lane_center - frame_center
        error_normalized = error / frame_center  # -1 a 1
        
        # PD simple para sugerir ángulo
        kp = 25.0  # Proporcional (máximo 25 grados)
        kd = 3.0   # Derivativo
        
        derivative = error_normalized - (self.previous_steering / kp if self.previous_steering else 0)
        steering_raw = kp * error_normalized + kd * derivative
        
        # Suavizar
        alpha = config.STEERING_SMOOTHING
        steering = alpha * steering_raw + (1 - alpha) * self.previous_steering
        
        # Clamp
        steering = max(-25.0, min(25.0, steering))
        self.previous_steering = steering
        
        return {
            'steering_angle': round(steering, 2),
            'lane_center_x': round(lane_center, 1),
            'confidence': round(confidence, 2),
            'error_normalized': round(error_normalized, 4),
        }
    
    @torch.no_grad()
    def infer(self, frame: np.ndarray) -> dict:
        """
        Ejecutar inferencia completa sobre un frame.
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            dict con todos los resultados:
              - 'steering': dict con ángulo de dirección sugerido
              - 'lane_mask': máscara de líneas comprimida (bytes)
              - 'road_mask': máscara de área manejable comprimida (bytes)
              - 'lane_points': puntos de las líneas detectadas
              - 'detections': lista de detecciones de objetos
              - 'inference_time_ms': tiempo de inferencia en ms
              - 'frame_id': ID del frame procesado
        """
        start_time = time.time()
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocesar
        input_tensor = self.preprocess(frame)
        
        # Inferencia
        try:
            outputs = self.model(input_tensor)
        except Exception as e:
            print(f"[HybridNets] ERROR en inferencia: {e}")
            return self._empty_result(orig_h, orig_w, start_time)
        
        # Log de estructura de salida (solo los primeros frames)
        if self.frame_count < 3:
            self._log_output_structure("outputs", outputs)
        
        # HybridNets devuelve una tupla de 5 elementos:
        #   outputs[0] = features del backbone (tupla de 5 tensores multi-escala) — no se usa
        #   outputs[1] = regression  [1, N, 4]   — bounding boxes
        #   outputs[2] = classification [1, N, 1] — scores de detección de objetos
        #   outputs[3] = anchors [1, N, 4]        — anchors para decodificar boxes
        #   outputs[4] = segmentation [1, 3, H, W] — mapa de segmentación (3 clases: bg, road, lane)
        
        regression = None
        classification = None
        seg_output = None
        
        try:
            if isinstance(outputs, tuple) and len(outputs) == 5:
                # Estructura estándar de HybridNets
                _features, regression, classification, _anchors, seg_output = outputs
                if self.frame_count < 3:
                    print(f"[HybridNets] Desempaquetado 5-tupla de HybridNets:")
                    print(f"  regression:     {regression.shape if hasattr(regression, 'shape') else type(regression)}")
                    print(f"  classification: {classification.shape if hasattr(classification, 'shape') else type(classification)}")
                    print(f"  seg_output:     {seg_output.shape if hasattr(seg_output, 'shape') else type(seg_output)}")
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                regression, classification, seg_output = outputs
                if self.frame_count < 3:
                    print(f"[HybridNets] Desempaquetado como (regression, classification, seg)")
            elif isinstance(outputs, dict):
                regression = outputs.get('regression')
                classification = outputs.get('classification')
                seg_output = outputs.get('segmentation', outputs.get('seg'))
                if self.frame_count < 3:
                    print(f"[HybridNets] Desempaquetado como dict, keys={list(outputs.keys())}")
            elif isinstance(outputs, torch.Tensor):
                # Tensor único — asumir segmentación directa
                seg_output = outputs
                if self.frame_count < 3:
                    print(f"[HybridNets] Salida es tensor directo: {outputs.shape}")
            else:
                # Fallback: buscar el tensor con forma [1, C, H, W] que parezca segmentación
                if isinstance(outputs, tuple):
                    for i, item in enumerate(outputs):
                        if isinstance(item, torch.Tensor) and item.dim() == 4:
                            h, w = item.shape[2], item.shape[3]
                            if h == self.input_height and w == self.input_width:
                                seg_output = item
                                if self.frame_count < 3:
                                    print(f"[HybridNets] Segmentación encontrada en outputs[{i}]: {item.shape}")
                                break
                if seg_output is None:
                    print(f"[HybridNets] WARN: No se pudo identificar la segmentación en outputs "
                          f"(tipo={type(outputs).__name__}, len={len(outputs) if hasattr(outputs, '__len__') else 'N/A'})")
                    return self._empty_result(orig_h, orig_w, start_time)
        except Exception as e:
            print(f"[HybridNets] WARN: Error desempaquetando salida: {e}")
            seg_output = None
        
        if seg_output is None:
            print(f"[HybridNets] WARN: seg_output es None, devolviendo resultado vacío")
            return self._empty_result(orig_h, orig_w, start_time)
        
        # Post-procesar segmentación
        seg_results = self.postprocess_segmentation(seg_output, (orig_h, orig_w))
        
        # Calcular dirección
        steering = self.compute_steering(
            seg_results['lane_points'], orig_w, orig_h
        )
        
        # Comprimir máscaras para transmisión eficiente
        lane_mask_compressed = self._compress_mask(seg_results['lane_mask'])
        road_mask_compressed = self._compress_mask(seg_results['road_mask'])
        
        # Post-procesar detecciones (si disponibles)
        detections = []
        if regression is not None and classification is not None:
            detections = self._postprocess_detections(
                regression, classification, orig_w, orig_h
            )
        
        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1
        
        return {
            'steering': steering,
            'lane_mask': lane_mask_compressed,
            'road_mask': road_mask_compressed,
            'lane_points': seg_results['lane_points'],
            'detections': detections,
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }
    
    @torch.no_grad()
    def infer_steering_only(self, frame: np.ndarray) -> dict:
        """
        Inferencia ligera: solo calcula segmentacion y steering.
        NO comprime mascaras ni post-procesa detecciones.
        Pensado para el endpoint /ws/steering donde solo se necesita el angulo.
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            dict con:
              - 'steering': dict con angulo de direccion sugerido
              - 'inference_time_ms': tiempo de inferencia en ms
              - 'frame_id': ID del frame procesado
              - 'input_size': resolucion de entrada del modelo
        """
        start_time = time.time()
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocesar
        t0 = time.time()
        input_tensor = self.preprocess(frame)
        t_preprocess = time.time()
        
        # Inferencia (forward pass del modelo)
        try:
            outputs = self.model(input_tensor)
        except Exception as e:
            print(f"[HybridNets] ERROR en inferencia: {e}")
            return self._empty_steering_result(start_time)
        t_forward = time.time()
        
        # Extraer segmentacion (misma logica que infer pero sin detecciones)
        seg_output = None
        
        try:
            if isinstance(outputs, tuple) and len(outputs) == 5:
                _features, _regression, _classification, _anchors, seg_output = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                _regression, _classification, seg_output = outputs
            elif isinstance(outputs, dict):
                seg_output = outputs.get('segmentation', outputs.get('seg'))
            elif isinstance(outputs, torch.Tensor):
                seg_output = outputs
            else:
                if isinstance(outputs, tuple):
                    for item in outputs:
                        if isinstance(item, torch.Tensor) and item.dim() == 4:
                            h, w = item.shape[2], item.shape[3]
                            if h == self.input_height and w == self.input_width:
                                seg_output = item
                                break
        except Exception as e:
            if self.frame_count < 3:
                print(f"[HybridNets] WARN: Error desempaquetando salida (steering_only): {e}")
            seg_output = None
        
        if seg_output is None:
            return self._empty_steering_result(start_time)
        
        # Post-procesar segmentacion (para lane points)
        seg_results = self.postprocess_segmentation(seg_output, (orig_h, orig_w))
        t_postprocess = time.time()
        
        # Calcular direccion
        steering = self.compute_steering(
            seg_results['lane_points'], orig_w, orig_h
        )
        t_steering = time.time()
        
        inference_ms = (t_steering - start_time) * 1000
        self.frame_count += 1
        
        # Log desglosado para los primeros 10 frames
        if self.frame_count <= 10:
            preproc_ms = (t_preprocess - t0) * 1000
            forward_ms = (t_forward - t_preprocess) * 1000
            postproc_ms = (t_postprocess - t_forward) * 1000
            steer_ms = (t_steering - t_postprocess) * 1000
            print(f"[Engine] Frame {self.frame_count}: "
                  f"preprocess={preproc_ms:.1f}ms | "
                  f"forward={forward_ms:.1f}ms | "
                  f"postprocess={postproc_ms:.1f}ms | "
                  f"steering={steer_ms:.1f}ms | "
                  f"TOTAL={inference_ms:.1f}ms | "
                  f"input={orig_w}x{orig_h} -> {self.input_width}x{self.input_height}")
        # Log resumido cada 100 frames
        elif self.frame_count % 100 == 0:
            print(f"[Engine] Frame {self.frame_count}: total={inference_ms:.1f}ms | "
                  f"steer={steering['steering_angle']}")
        
        return {
            'steering': steering,
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }
    
    def _empty_steering_result(self, start_time):
        """Resultado vacio para infer_steering_only cuando falla la inferencia."""
        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1
        return {
            'steering': {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            },
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }

    def _compress_mask(self, mask: np.ndarray) -> bytes:
        """Comprimir máscara binaria usando PNG para transmisión eficiente."""
        # Reducir resolución para ahorrar ancho de banda
        small = cv2.resize(mask, (self.input_width // 2, self.input_height // 2),
                           interpolation=cv2.INTER_NEAREST)
        _, encoded = cv2.imencode('.png', small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return encoded.tobytes()
    
    def _postprocess_detections(self, regression, classification, orig_w, orig_h):
        """Post-procesar detecciones de objetos."""
        detections = []
        try:
            # Aplicar NMS y filtrar por confianza
            scores = torch.sigmoid(classification).squeeze(0)  # [num_anchors, num_classes]
            
            if scores.dim() < 2:
                return detections
            
            max_scores, class_ids = scores.max(dim=1)
            
            # Filtrar por confianza
            mask = max_scores > config.CONF_THRESHOLD
            if mask.sum() == 0:
                return detections
            
            filtered_scores = max_scores[mask]
            filtered_classes = class_ids[mask]
            filtered_boxes = regression.squeeze(0)[mask]
            
            # Convertir a formato [x1, y1, x2, y2] si es necesario
            # (depende del formato de salida del modelo)
            
            for i in range(len(filtered_scores)):
                det = {
                    'class_id': int(filtered_classes[i]),
                    'class_name': config.DET_CLASSES[min(int(filtered_classes[i]), len(config.DET_CLASSES)-1)],
                    'confidence': round(float(filtered_scores[i]), 3),
                    'bbox': filtered_boxes[i].cpu().numpy().tolist(),
                }
                detections.append(det)
        except Exception as e:
            print(f"[HybridNets] WARN: Error en post-procesado de detecciones: {e}")
        
        return detections
    
    def _log_output_structure(self, name: str, obj, depth: int = 0):
        """Log recursivo de la estructura de un output del modelo."""
        indent = "  " * depth
        if isinstance(obj, torch.Tensor):
            print(f"{indent}[HybridNets] {name}: Tensor shape={obj.shape}, dtype={obj.dtype}, device={obj.device}")
        elif isinstance(obj, np.ndarray):
            print(f"{indent}[HybridNets] {name}: ndarray shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, (tuple, list)):
            kind = "tuple" if isinstance(obj, tuple) else "list"
            print(f"{indent}[HybridNets] {name}: {kind} len={len(obj)}")
            if depth < 2:  # No profundizar demasiado
                for i, item in enumerate(obj):
                    self._log_output_structure(f"{name}[{i}]", item, depth + 1)
        elif isinstance(obj, dict):
            print(f"{indent}[HybridNets] {name}: dict keys={list(obj.keys())}")
        else:
            print(f"{indent}[HybridNets] {name}: {type(obj).__name__}")

    def _empty_result(self, orig_h, orig_w, start_time):
        """Resultado vacío cuando falla la inferencia."""
        inference_ms = (time.time() - start_time) * 1000
        self.frame_count += 1
        return {
            'steering': {
                'steering_angle': None,
                'lane_center_x': None,
                'confidence': 0.0,
                'error_normalized': 0.0,
            },
            'lane_mask': b'',
            'road_mask': b'',
            'lane_points': [],
            'detections': [],
            'inference_time_ms': round(inference_ms, 1),
            'frame_id': self.frame_count,
            'input_size': f"{self.input_width}x{self.input_height}",
        }
    
    def get_status(self) -> dict:
        """Obtener estado del motor de inferencia."""
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated_mb': round(torch.cuda.memory_allocated(0) / 1024**2, 1),
                'memory_total_mb': round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1),
            }
        
        return {
            'model_loaded': self.model is not None,
            'device': self.device_str,
            'use_half': self.use_half,
            'input_size': f"{self.input_width}x{self.input_height}",
            'frames_processed': self.frame_count,
            'gpu_info': gpu_info,
        }
