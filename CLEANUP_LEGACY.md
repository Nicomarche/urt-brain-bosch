# Limpieza de Codigo Legacy - URT Brain Bosch

> **Estado actual:** Solo se usa `AI_LOCAL` (YOLO TensorRT 416px).  
> Todo lo demas (LSTR, OpenCV, HybridNets, Supercombo, remote sign detection) es legacy y puede eliminarse.
>
> **SE MANTIENEN:** `src/data/Semaphores/`, `src/data/TrafficCommunication/`, `SystemMode.LEGACY` y el boton legacy del dashboard (se van a usar en el futuro).

---

## 1. ARCHIVOS COMPLETOS A ELIMINAR

### 1.1 `src/hardware/camera/threads/lstrDetector.py` (~554 lineas)

**Que es:** Detector de carriles LSTR (Lane Shape Prediction with Transformers) basado en ONNX Runtime.

**Contenido:**
- Clase `LSTRModelType(Enum)` - 5 tamaños de modelo (180x320 a 720x1280)
- Clase `LSTRDetector` - inferencia ONNX, deteccion de carriles, estimacion de curvatura
- Metodos: `detect_lanes()`, `get_lane_center()`, `estimate_path_curvature()`, `draw_lanes()`

**Dependencia:** Importado SOLO por `threadLineFollowing.py:32` dentro de `try/except ImportError`:
```python
try:
    from src.hardware.camera.threads.lstrDetector import LSTRDetector, LSTRModelType
    LSTR_AVAILABLE = True
except ImportError as e:
    LSTR_AVAILABLE = False
```
Al eliminar el archivo, `LSTR_AVAILABLE = False` y todo sigue funcionando.

---

### 1.2 `src/hardware/camera/threads/signDetector.py` (~186 lineas)

**Que es:** Detector de senales TFLite (MobilenetV2 SSD) standalone. Reemplazado completamente por YOLO local.

**Contenido:**
- Clase `SignDetector` - wrapper de TFLite con runtime fallback (ai_edge_litert / tflite_runtime / tensorflow)
- Metodos: `detect()`, `detect_all()`
- Modelo: `models/sign_detection/detect.tflite`

**Dependencia:** **NINGUNA** - ningun archivo del proyecto lo importa.

---

### 1.3 `src/hardware/camera/threads/threadSignDetection.py` (~734 lineas)

**Que es:** Thread de deteccion de senales remoto via WebSocket a un AI Server. Contiene su propia copia de `SignActions`.

**Contenido:**
- Clase `SignDetectionClient` - cliente WebSocket que envia JPEGs al servidor
- Clase `SignActions` (duplicada) - ejecutor de acciones de senales
- Clase `threadSignDetection(ThreadWithStop)` - thread principal

**Dependencia:** Importado SOLO por `processCamera.py:50` dentro de `try/except ImportError`:
```python
try:
    from src.hardware.camera.threads.threadSignDetection import threadSignDetection
    SIGN_DETECTION_AVAILABLE = True
except ImportError:
    SIGN_DETECTION_AVAILABLE = False
```
Ademas, solo se instancia cuando `self.use_legacy_remote_sign_detection = True` (`processCamera.py:117`), que esta hardcodeado a `False`.

> **ATENCION:** `signActions.py` (sin el "thread") **NO se elimina**. Lo usan activamente `maneuverManager.py:8` y `threadLocalPerception.py:8`.

---

### 1.4 Directorio `aiserver/` (entero)

**Que es:** Servidor de inferencia remoto para HybridNets, Supercombo y YOLO lane seg. Ya no se usa.

**Contenido:**
| Archivo | Descripcion |
|---------|------------|
| `server.py` | WebSocket server |
| `client.py` | Cliente WebSocket (HybridNetsClient) |
| `inference.py` | Orquestacion de inferencia |
| `supercombo_engine.py` | Motor openpilot supercombo |
| `yolo_lane_seg_engine.py` | YOLO lane seg server-side |
| `sign_detection_engine.py` | Deteccion de senales TFLite |
| `config.py` | Configuracion del server |
| `setup_server.py` | Setup del server |
| `setup_supercombo.py` | Setup del supercombo |
| `HybridNets/` | Modelo HybridNets completo + encoders + training |
| `models/` | Archivos de modelos |
| `weights/` | Pesos pre-entrenados |
| `venv/` | Entorno virtual del server |

**Dependencia:** `aiserver.client.HybridNetsClient` importado por `threadLineFollowing.py:40` dentro de `try/except`:
```python
try:
    from aiserver.client import HybridNetsClient
    HYBRIDNETS_CLIENT_AVAILABLE = True
except ImportError:
    HYBRIDNETS_CLIENT_AVAILABLE = False
```
Tambien `localPerceptionEngine.py:195` tiene un fallback path a `aiserver/models/lane_segmentation/` que retorna `False` si no existe el directorio.

---

## 2. CODIGO LEGACY DENTRO DE ARCHIVOS ACTIVOS

### 2.1 `threadLineFollowing.py` - LA LIMPIEZA MAS GRANDE

Este archivo tiene ~12,778 lineas. Se puede remover una cantidad significativa de codigo legacy.

#### 2.1.a Imports legacy (lineas 30-44)

**Eliminar bloque LSTR (lineas 30-36):**
```python
try:
    from src.hardware.camera.threads.lstrDetector import LSTRDetector, LSTRModelType
    LSTR_AVAILABLE = True
except ImportError as e:
    LSTR_AVAILABLE = False
    print(...)
```

**Eliminar bloque HybridNets (lineas 38-44):**
```python
try:
    from aiserver.client import HybridNetsClient
    HYBRIDNETS_CLIENT_AVAILABLE = True
except ImportError as e:
    HYBRIDNETS_CLIENT_AVAILABLE = False
    print(...)
```

#### 2.1.b Enum `DetectionMode` (lineas 46-54)

**Estado actual:**
```python
class DetectionMode(Enum):
    OPENCV = "opencv"          # ELIMINAR
    LSTR = "lstr"              # ELIMINAR
    HYBRID = "hybrid"          # ELIMINAR
    AI_LOCAL = "ai_local"      # MANTENER (unico modo activo)
    HYBRIDNETS = "hybridnets"  # ELIMINAR (alias deprecated)
    SUPERCOMBO = "supercombo"  # ELIMINAR (alias deprecated)
```

**Despues de limpieza:** Dejar solo `AI_LOCAL` o eliminar el enum completamente y hardcodear el string.

#### 2.1.c Atributos legacy en `__init__` 

**Eliminar atributos LSTR (~lineas 987-992):**
- `lstr_model_size`, `lstr_detector`, `lstr_fallback_threshold`, `lstr_confidence_threshold`, `_current_lstr_model_size`

**Eliminar atributos Hybrid (~lineas 994-997):**
- `hybrid_opencv_weight`, `hybrid_lstr_weight`, `hybrid_agreement_bonus`

**Eliminar atributos HybridNets (~lineas 999-1003):**
- `hybridnets_server_url`, `hybridnets_jpeg_quality`, `hybridnets_timeout`, `hybridnets_max_result_age`
- `_hybridnets_client` (linea 1024)

**Eliminar atributos Supercombo (~lineas 1113-1117):**
- `supercombo_server_url`, `supercombo_jpeg_quality`, `supercombo_timeout`, `_supercombo_client`

**Eliminar llamada a init LSTR (linea 1119):**
- `self._init_lstr_detector()`

#### 2.1.d Metodos legacy completos a eliminar

| Metodo | Lineas aprox. | LOC | Que hace |
|--------|--------------|-----|----------|
| `_init_lstr_detector()` | 1343-1377 | ~35 | Inicializa detector LSTR ONNX |
| `_init_hybridnets_client()` | 1379-1395 | ~17 | Inicializa cliente WebSocket HybridNets |
| `_resolve_hybridnets_inference_url()` | 1397-1404 | ~8 | Resuelve URL del server |
| `_normalize_detection_mode()` | 1406-1411 | ~6 | Mapea HYBRIDNETS/SUPERCOMBO a AI_LOCAL |
| `_start_hybridnets_client()` | 2876-2890 | ~15 | Conecta cliente HybridNets |
| `_stop_hybridnets_client()` | 2892-2898 | ~7 | Desconecta cliente HybridNets |
| `_init_supercombo_client()` | 2904-2920 | ~17 | Inicializa cliente Supercombo |
| `_start_supercombo_client()` | 2922-2935 | ~14 | Conecta cliente Supercombo |
| `_stop_supercombo_client()` | 2937-2943 | ~7 | Desconecta cliente Supercombo |
| `_detect_with_supercombo()` | 2945-3069 | ~125 | Pipeline completo de deteccion Supercombo |
| `_detect_with_hybridnets()` | 3071-3170 | ~100 | Pipeline completo de deteccion HybridNets |
| `_detect_with_lstr()` | 5762-5935 | ~174 | Pipeline completo de deteccion LSTR |
| `_detect_hybrid_fusion()` | 5938-6112+ | ~175 | Fusion OpenCV + LSTR |

**Funciones de procesamiento OpenCV (solo usadas por path OpenCV/BFMC):**

| Funcion | Lineas aprox. | LOC | Que hace |
|---------|--------------|-----|----------|
| `_apply_clahe()` | ~5495 | ~15 | Contrast Limited Adaptive Histogram Equalization |
| `_adaptive_white_detection()` | ~5510 | ~30 | Deteccion adaptativa de blanco HSV |
| `_gradient_based_detection()` | ~5540 | ~25 | Deteccion por gradiente |
| `_preprocess_frame()` | ~5670 | ~40 | Preprocesamiento de frame para OpenCV |
| `_create_combined_mask()` | ~5710 | ~50 | Mascara combinada HSV + bordes |
| `_filter_line_like_components()` | ~5760 | ~30 | Filtro de componentes tipo linea |
| `_adaptive_local_threshold()` | ~5790 | ~20 | Threshold local adaptativo |
| `_hough_lines_by_half()` | ~5810 | ~30 | Deteccion Hough dividida por mitades |
| `_bfmc_image_processing()` | ~7703-7802 | ~100 | Pipeline completo BFMC (threshold+canny+hough) |
| `_bfmc_classify_lines()` | ~7803+ | ~50 | Clasificacion de lineas BFMC |

> **NOTA:** Verificar que `_update_stopline_visual_state()` NO use estas funciones OpenCV. Usa su propia deteccion BEV independiente, asi que es seguro eliminarlas.

#### 2.1.e Simplificar `process_frame()` (~lineas 11057-11249)

**Eliminar estos bloques del metodo:**

- Lineas ~11091-11113: Bloque `if self.detection_mode == DetectionMode.SUPERCOMBO.value`
- Lineas ~11116-11138: Bloque `if self.detection_mode == DetectionMode.HYBRID.value`
- Lineas ~11141-11179: Bloque `if self.detection_mode == DetectionMode.LSTR.value`
- Lineas ~11181-11188: Simplificar `using_remote_lanes` (siempre es True con AI_LOCAL)
- Lineas ~11200-11233: Eliminar rama `else` de OpenCV (el `if using_remote_lanes` siempre es True)

**Despues de limpieza:** `process_frame()` siempre llama `_detect_with_local_ai()`.

#### 2.1.f Limpiar config handler (~lineas 9570-9868)

**Eliminar de `string_params`:**
- `'hybridnets_server_url'`
- `'supercombo_server_url'`

**Eliminar de `int_params`:**
- `'lstr_model_size'`
- `'hybridnets_jpeg_quality'`
- `'supercombo_jpeg_quality'`
- Parametros OpenCV: `'blur_kernel'`, `'morph_kernel'`, `'canny_low'`, `'canny_high'`, `'hough_threshold'`, etc.

**Eliminar bloques handler:**
- Lineas ~9837-9839: Reload modelo LSTR
- Lineas ~9841-9844: Reconexion HybridNets
- Lineas ~9846-9849: Reconexion Supercombo
- Lineas ~9851-9861: Start/stop de clientes remotos segun modo

#### 2.1.g Limpiar debug panel/status (~lineas 5190-5430)

- Eliminar colores de modos legacy en `mode_colors` dict (`'opencv'`, `'lstr'`, `'hybrid'`)
- Eliminar display de status LSTR (`lstr_available`)
- Eliminar LSTR del mapping de `stream_debug_view`

---

### 2.2 `processCamera.py`

**Eliminar:**

| Linea | Contenido |
|-------|-----------|
| 50-54 | `try/except` import de `threadSignDetection` y flag `SIGN_DETECTION_AVAILABLE` |
| 93 | Parametro `sign_server_url` del `__init__` |
| 116 | `self.sign_server_url = sign_server_url` |
| 117 | `self.use_legacy_remote_sign_detection = False` |
| 291-309 | Bloque completo `if self.use_legacy_remote_sign_detection` (creacion de `signDetTh`) |

---

### 2.3 `localPerceptionEngine.py`

**Eliminar (~linea 195):** Fallback path a `aiserver/models/`:
```python
legacy = os.path.join(repo_root, "aiserver", "models", "lane_segmentation", os.path.basename(model_path))
if os.path.isfile(legacy):
    return legacy
```

---

### 2.4 `config.py`

**Eliminar:**

| Linea | Contenido |
|-------|-----------|
| 570 | `"ai_analysis": False,  # Analisis de LSTR / AI` |
| 571 | `"hybrid_fusion": False,  # Fusion hibrida OpenCV + LSTR` |
| 583-585 | `SIGN_SERVER_URL = "ws://localhost:8500/ws/signs"` y sus comentarios legacy |

**Actualizar:**

| Linea | Cambio |
|-------|--------|
| 577-579 | Reescribir header de seccion SIGN DETECTION para que no mencione "AI Server remoto" ni "WebSocket" |

---

### 2.5 `main.py`

**No tocar** las lineas 65-66 (imports de Semaphores y TrafficCommunication) ni las lineas comentadas 179, 183, 227-228. Se van a usar en el futuro.

---

## 3. DASHBOARD FRONTEND CLEANUP (DETALLADO)

### 3.1 `state-switch.component.ts` - NO TOCAR

Se mantiene `'legacy'` en el array de estados (linea 44) y el color en `getSliderColor()` (linea 343) porque Semaphores/TrafficCommunication se van a usar.

---

### 3.2 `line-following.component.html` - Limpieza de UI

#### Eliminar: Status LSTR (lineas 64-69)
```html
<!-- ELIMINAR este bloque completo -->
<div class="stat-item" *ngIf="selectedMode !== 'ai_local'">
  <span class="stat-label">LSTR</span>
  <span class="stat-value" [class.available]="debugStatus?.lstr_available">
    {{ debugStatus?.lstr_available ? 'Disponible' : 'No disponible' }}
  </span>
</div>
```

#### Eliminar: Botones de modo OpenCV, LSTR, Hibrido (lineas 92-119)
Eliminar los 3 botones legacy, mantener solo AI Local (lineas 120-131). El `<div class="button-group mode-buttons">` se puede simplificar o incluso eliminar la seccion entera de "Modo de Deteccion" si solo queda un modo.

```html
<!-- ELIMINAR estos 3 botones (lineas 92-119) -->
<button class="mode-btn" [class.selected]="selectedMode === 'opencv'" (click)="setMode('opencv')">
  <!-- OpenCV -->
</button>
<button class="mode-btn" [class.selected]="selectedMode === 'lstr'" ...>
  <!-- LSTR AI -->
</button>
<button class="mode-btn" [class.selected]="selectedMode === 'hybrid'" ...>
  <!-- Hibrido -->
</button>

<!-- MANTENER solo el boton AI Local (lineas 120-131) -->
```

#### Eliminar: Seccion seleccion de modelo LSTR (lineas 188-205)
```html
<!-- ELIMINAR bloque completo -->
<div class="config-section" *ngIf="selectedMode === 'lstr' || selectedMode === 'hybrid'">
  <div class="section-header"><h4>Modelo LSTR</h4></div>
  <!-- ... seleccion de modelo con 5 botones ... -->
</div>
```

#### Eliminar: Seccion PID (lineas 277-292)
Gateada por `isSectionVisibleForMode('pid')` que incluye opencv/lstr/hybrid/ai_local. **OJO:** Revisar si PID se usa en AI_LOCAL. Si el PID solo aplica en modo OpenCV y AI_LOCAL usa Stanley, eliminar. Si se usa en ambos, mantener.

#### Eliminar: Seccion Feed-Forward (lineas 294-309)
Gateada por `isSectionVisibleForMode('feedforward')` que incluye opencv/lstr/hybrid/ai_local. Misma logica que PID.

#### Eliminar: Seccion ROI (lineas 311-326)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('roi')">
  <h4>Region de Interes (ROI)</h4>
  <!-- roi_height_start, roi_height_end, roi_width_margin_top, roi_width_margin_bottom -->
</div>
```

#### Eliminar: Seccion Procesamiento de Imagen (lineas 328-343)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('image')">
  <h4>Procesamiento de Imagen</h4>
  <!-- brightness, contrast, blur_kernel, morph_kernel -->
</div>
```

#### Eliminar: Seccion Linea Blanca HSV (lineas 345-360)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('white')">
  <h4>Linea Blanca (HSV)</h4>
  <!-- white_h_min/max, white_s_min/max, white_v_min/max -->
</div>
```

#### Eliminar: Seccion Linea Amarilla HSV (lineas 362-377)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('yellow')">
  <h4>Linea Amarilla (HSV)</h4>
  <!-- yellow_h_min/max, yellow_s_min/max, yellow_v_min/max -->
</div>
```

#### Eliminar: Seccion Deteccion de Bordes (lineas 379-394)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('edge')">
  <h4>Deteccion de Bordes</h4>
  <!-- canny_low, canny_high, hough_threshold, hough_min_line_length, hough_max_line_gap -->
</div>
```

#### Eliminar: Seccion Parametros BFMC (lineas 396-411)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('bfmc')">
  <h4>Parametros BFMC</h4>
  <!-- binary_threshold, binary_threshold_retry, line_angle_filter, line_merge_distance -->
</div>
```

#### Eliminar: Seccion Iluminacion Adaptativa (lineas 413-446)
```html
<!-- ELIMINAR - solo visible en opencv/hybrid -->
<div class="config-section" *ngIf="isSectionVisibleForMode('adaptive')">
  <h4>Iluminacion Adaptativa</h4>
  <!-- CLAHE toggle, Blanco Adaptativo toggle, Gradiente Fallback toggle + sliders -->
</div>
```

#### Eliminar: Hint de modo LSTR (lineas 476-478)
```html
<!-- ELIMINAR -->
<div class="config-section remote-mode-hint" *ngIf="selectedMode === 'lstr'">
  <p>En modo LSTR AI, la red neuronal detecta los carriles directamente...</p>
</div>
```

---

### 3.3 `line-following.component.ts` - Limpieza de logica

#### Eliminar: Interface `LstrModel` (lineas 23-28)
```typescript
// ELIMINAR
interface LstrModel {
  id: number;
  name: string;
  resolution: string;
  speed: string;
}
```

#### Eliminar del interface `DebugStatus` (lineas 36, 54-59)
```typescript
// ELIMINAR estas propiedades:
lstr_available: boolean;              // linea 36
hybridnets_connected?: boolean;       // linea 54
hybridnets_roundtrip_ms?: number;     // linea 55
hybridnets_server_fps?: number;       // linea 56
supercombo_connected?: boolean;       // linea 57
supercombo_roundtrip_ms?: number;     // linea 58
supercombo_server_fps?: number;       // linea 59
```

#### Cambiar default mode (linea 72)
```typescript
// ANTES:
selectedMode: string = 'opencv';
// DESPUES:
selectedMode: string = 'ai_local';
```

#### Eliminar: Propiedad y array LSTR (lineas 73-83)
```typescript
// ELIMINAR
lstrAvailable: boolean = true;
selectedLstrModel: number = 0;
lstrModels: LstrModel[] = [
  { id: 0, name: 'Ultra Rapido', resolution: '180x320', speed: '~15 FPS' },
  { id: 1, name: 'Rapido', resolution: '240x320', speed: '~12 FPS' },
  { id: 2, name: 'Balanceado', resolution: '360x640', speed: '~8 FPS' },
  { id: 3, name: 'Preciso', resolution: '480x640', speed: '~5 FPS' },
  { id: 4, name: 'Maxima Calidad', resolution: '720x1280', speed: '~2 FPS' },
];
```

#### Eliminar: Propiedades HybridNets (lineas 85-91)
```typescript
// ELIMINAR
hybridnetsServerUrl: string = 'ws://127.0.0.1:8500/ws/steering';
hybridnetsJpegQuality: number = 70;
hybridnetsTimeout: number = 2.0;
hybridnetsConnected: boolean = false;
hybridnetsRoundtripMs: number = 0;
hybridnetsServerFps: number = 0;
```

#### Eliminar: Propiedades Supercombo (lineas 101-107)
```typescript
// ELIMINAR
supercomboServerUrl: string = 'ws://127.0.0.1:8500/ws/steering';
supercomboJpegQuality: number = 70;
supercomboTimeout: number = 2.0;
supercomboConnected: boolean = false;
supercomboRoundtripMs: number = 0;
supercomboServerFps: number = 0;
```

#### Eliminar: Vista de debug LSTR IA (linea 123 dentro de `debugViews`)
```typescript
// ELIMINAR esta entrada del array debugViews:
{ id: 10, name: 'LSTR IA', icon: '🤖', description: 'Salida del modelo de IA LSTR con carriles detectados.' },
```

#### Eliminar: Toggles y sliders de OpenCV (lineas 131-134, 140-152)
```typescript
// ELIMINAR toggles:
useClahe: boolean = true;
useAdaptiveWhite: boolean = true;
useGradientFallback: boolean = true;

// ELIMINAR de expandedSections (dejar solo los que aplican a AI_LOCAL):
// Eliminar: pid, feedforward, roi, white, yellow, image, edge, bfmc, adaptive
// Mantener: speed, recovery (si aplica)
```

#### Eliminar: Sliders de OpenCV (dentro del array `sliders`, lineas 155-225)
Eliminar todos los sliders de estos grupos:
- `group: 'pid'` (lineas 161-169) - 9 sliders
- `group: 'feedforward'` (lineas 171-173) - 3 sliders
- `group: 'roi'` (lineas 175-178) - 4 sliders
- `group: 'white'` (lineas 180-185) - 6 sliders
- `group: 'yellow'` (lineas 187-192) - 6 sliders
- `group: 'image'` (lineas 194-197) - 4 sliders
- `group: 'edge'` (lineas 199-203) - 5 sliders
- `group: 'bfmc'` (lineas 205-208) - 4 sliders
- `group: 'adaptive'` (lineas 210-215) - 6 sliders

**Mantener:**
- `group: 'speed'` (lineas 157-159) - 3 sliders
- `group: 'recovery'` (lineas 217-224) - 8 sliders (si aplica en AI_LOCAL)

> **OJO:** Revisar si los sliders de `speed` y `recovery` se usan en AI_LOCAL antes de decidir. Si speed se controla desde otro lugar en AI_LOCAL, tambien se puede sacar.

#### Simplificar o eliminar: `normalizeMode()` (lineas 243-248)
```typescript
// ANTES:
private normalizeMode(mode: string | null | undefined): string {
  if (mode === 'hybridnets' || mode === 'supercombo') {
    return 'ai_local';
  }
  return mode || 'opencv';
}
// DESPUES: eliminar y reemplazar por un return directo 'ai_local'
// o simplificar a: return 'ai_local';
```

#### Limpiar: `ngOnInit()` subscription a status (lineas 276-301)
```typescript
// ELIMINAR estas lineas del subscribe:
this.lstrAvailable = status?.lstr_available ?? false;          // linea 277
// Update HybridNets connection status (lineas 282-290)
if (status?.hybridnets_connected !== undefined) { ... }
if (status?.hybridnets_roundtrip_ms !== undefined) { ... }
if (status?.hybridnets_server_fps !== undefined) { ... }
// Update Supercombo connection status (lineas 292-300)
if (status?.supercombo_connected !== undefined) { ... }
if (status?.supercombo_roundtrip_ms !== undefined) { ... }
if (status?.supercombo_server_fps !== undefined) { ... }
```

#### Simplificar: `setMode()` (linea 317-321)
```typescript
// ANTES:
setMode(mode: string): void {
  mode = this.normalizeMode(mode);
  if ((mode === 'lstr' || mode === 'hybrid') && !this.lstrAvailable) return;
  this.selectedMode = mode;
  this.debouncedSendConfig();
}
// DESPUES: eliminar guard de LSTR/hybrid, simplificar
```

#### Simplificar: `getModeDisplayName()` (lineas 324-333)
```typescript
// ANTES:
const names: { [key: string]: string } = {
  'opencv': 'OpenCV',           // ELIMINAR
  'lstr': 'LSTR IA',           // ELIMINAR
  'hybrid': 'Hibrido',         // ELIMINAR
  'ai_local': 'AI Local',      // MANTENER
  'hybridnets': 'HybridNets',  // ELIMINAR
  'supercombo': 'Supercombo'   // ELIMINAR
};
// DESPUES: return 'AI Local';
```

#### Eliminar: Todos los metodos HybridNets (lineas 369-389)
```typescript
// ELIMINAR estos 4 metodos:
setHybridnetsServerUrl(url: string): void { ... }
setHybridnetsJpegQuality(quality: number): void { ... }
setHybridnetsTimeout(timeout: number): void { ... }
setHybridnetsEndpoint(endpoint: string): void { ... }
```

#### Eliminar: Todos los metodos Supercombo (lineas 391-411)
```typescript
// ELIMINAR estos 4 metodos:
setSupercomboServerUrl(url: string): void { ... }
setSupercomboJpegQuality(quality: number): void { ... }
setSupercomboTimeout(timeout: number): void { ... }
setSupercomboEndpoint(endpoint: string): void { ... }
```

#### Eliminar: Metodo LSTR (lineas 413-417)
```typescript
// ELIMINAR
setLstrModel(modelId: number): void { ... }
```

#### Eliminar: Toggles de OpenCV (lineas 450-463)
```typescript
// ELIMINAR estos 3 metodos:
toggleClahe(): void { ... }
toggleAdaptiveWhite(): void { ... }
toggleGradientFallback(): void { ... }
```

#### Simplificar: `isSectionVisibleForMode()` (lineas 476-494)
```typescript
// ANTES: chequea opencv/lstr/hybrid/ai_local
// DESPUES: solo necesita saber que secciones aplican a ai_local
// Eliminar las referencias a 'opencv', 'lstr', 'hybrid'
// Las secciones de opencvSections (roi, white, yellow, etc.) nunca se muestran -> eliminar
```

#### Limpiar: `sendConfig()` (lineas 516-553)
```typescript
// ELIMINAR de config:
config['lstr_model_size'] = this.selectedLstrModel;  // linea 521
// Tambien los sliders de OpenCV se dejan de enviar al eliminarlos del array
```

#### Limpiar: `applyConfig()` (lineas 555-629)
```typescript
// ELIMINAR:
// lstr_model_size apply (lineas 560-562)
if (config['lstr_model_size'] !== undefined) {
  this.selectedLstrModel = config['lstr_model_size'];
}
// HybridNets settings (lineas 573-582)
if (config['hybridnets_server_url']) { ... }
if (config['hybridnets_jpeg_quality'] !== undefined) { ... }
if (config['hybridnets_timeout'] !== undefined) { ... }
// Supercombo settings (lineas 584-593)
if (config['supercombo_server_url']) { ... }
if (config['supercombo_jpeg_quality'] !== undefined) { ... }
if (config['supercombo_timeout'] !== undefined) { ... }
// Toggles de OpenCV (lineas 610-621)
if (config['use_clahe'] !== undefined) { ... }
if (config['use_adaptive_white'] !== undefined) { ... }
if (config['use_gradient_fallback'] !== undefined) { ... }
```

#### Limpiar: `resetDefaults()` (lineas 631-689)
```typescript
// CAMBIAR default mode (linea 633):
this.selectedMode = 'ai_local';  // era 'opencv'

// ELIMINAR (linea 634):
this.selectedLstrModel = 0;

// ELIMINAR reset HybridNets (lineas 639-642):
this.hybridnetsServerUrl = 'ws://127.0.0.1:8500/ws/steering';
this.hybridnetsJpegQuality = 70;
this.hybridnetsTimeout = 2.0;

// ELIMINAR reset Supercombo (lineas 644-647):
this.supercomboServerUrl = 'ws://127.0.0.1:8500/ws/steering';
this.supercomboJpegQuality = 70;
this.supercomboTimeout = 2.0;

// ELIMINAR toggles OpenCV del reset (lineas 656-658):
this.useClahe = true;
this.useAdaptiveWhite = true;
this.useGradientFallback = true;

// ELIMINAR del objeto defaults (lineas 661-676):
// Todos los keys de: pid, feedforward, roi, white, yellow, image, edge, bfmc, adaptive
// Mantener solo: speed + recovery (si aplican)
```

---

### 3.4 `line-following.component.css` - Limpieza de estilos

#### Eliminar: Estilos de botones de modelo LSTR (lineas 288-334)
```css
/* ELIMINAR - Model Buttons */
.model-buttons { ... }
.model-btn { ... }
.model-btn:hover { ... }
.model-btn.selected { ... }
.model-name { ... }
.model-resolution { ... }
.model-speed { ... }
```

#### Simplificar: Grid de mode-buttons (linea 226)
```css
/* ANTES: 5 columnas para 5 botones */
.mode-buttons {
  grid-template-columns: repeat(5, 1fr);
}
/* DESPUES: si queda solo 1 boton, ajustar o eliminar el grid */
.mode-buttons {
  grid-template-columns: 1fr;  /* o eliminar la seccion entera */
}
```

#### Opcional: Renombrar clases "hybridnets" (lineas 611-755)
Estas clases se usan para estilizar la seccion de AI Local (reutilizaron los nombres):
```css
.hybridnets-btn.selected { ... }     /* linea 612 - boton AI Local seleccionado */
.hybridnets-config { ... }           /* linea 627 - seccion config AI Local */
.hybridnets-hint { ... }             /* linea 738 - hint de AI Local */
.hybridnets-hint code { ... }        /* linea 749 */
```
**Recomendacion:** Renombrar a `.ai-local-btn.selected`, `.ai-local-config`, `.ai-local-hint` para que el CSS refleje el estado actual.

#### Actualizar: Responsive breakpoints (lineas 773-826)
```css
/* Actualizar grids en media queries */
@media (max-width: 900px) {
  .mode-buttons {
    grid-template-columns: repeat(3, 1fr);  /* AJUSTAR o eliminar */
  }
  .model-buttons {
    grid-template-columns: repeat(3, 1fr);  /* ELIMINAR - ya no hay model-buttons */
  }
}
@media (max-width: 600px) {
  .mode-buttons {
    grid-template-columns: repeat(2, 1fr);  /* AJUSTAR o eliminar */
  }
  .model-buttons {
    grid-template-columns: repeat(2, 1fr);  /* ELIMINAR */
  }
}
```

---

## 4. ORDEN DE OPERACIONES SEGURO

Ejecutar en este orden para minimizar riesgo:

### Fase 1 - Archivos standalone (riesgo bajo, hay try/except)
1. Eliminar `src/hardware/camera/threads/lstrDetector.py`
2. Eliminar `src/hardware/camera/threads/signDetector.py`
3. Eliminar `src/hardware/camera/threads/threadSignDetection.py`
4. **Probar** - el sistema debe arrancar sin errores

### Fase 2 - Limpiar imports y referencias en archivos activos
5. Limpiar `processCamera.py` (seccion 2.2)
6. Limpiar `localPerceptionEngine.py` (seccion 2.3)
7. Eliminar directorio `aiserver/`
8. **Probar** - el sistema debe arrancar sin errores

### Fase 3 - Limpieza grande de threadLineFollowing.py
9. Eliminar imports legacy (2.1.a)
10. Simplificar enum DetectionMode (2.1.b)
11. Eliminar atributos legacy del __init__ (2.1.c)
12. Eliminar metodos legacy (2.1.d) - empezar por los mas aislados
13. Simplificar process_frame() (2.1.e)
14. Limpiar config handler (2.1.f)
15. Limpiar debug panel (2.1.g)
16. **Probar** - validar que AI_LOCAL sigue funcionando correctamente

### Fase 4 - Config
17. Limpiar `config.py` (seccion 2.4)
18. **Probar**

### Fase 5 - Dashboard frontend
19. Limpiar `line-following.component.ts` (seccion 3.3)
20. Limpiar `line-following.component.html` (seccion 3.2)
21. Limpiar `line-following.component.css` (seccion 3.4)
22. Rebuild frontend: `cd src/dashboard/frontend && ng build`
23. **Probar** - verificar UI del dashboard

---

## 5. VERIFICACION POST-LIMPIEZA

### Smoke tests manuales:
- [ ] Sistema arranca y entra en modo DEFAULT
- [ ] Cambio de modo desde dashboard: DEFAULT -> AUTO -> MANUAL -> LEGACY -> PARKING -> STOP
- [ ] AI_LOCAL corre: la camara captura frames, YOLO produce masks y detecciones
- [ ] Steering commands se envian correctamente (Stanley controller)
- [ ] Acciones de senales funcionan (stop, highway, crosswalk, etc.)
- [ ] Parking maneuver FSM funciona
- [ ] Tracking/navigation funciona (dead reckoning, waypoints)
- [ ] Settings de line-following solo muestra AI Local, sin botones OpenCV/LSTR/Hybrid
- [ ] Secciones de config OpenCV (ROI, HSV, Canny, BFMC) ya no aparecen en el dashboard

### Grep de validacion (no debe encontrar nada excepto en este archivo):
```bash
grep -r "LSTRDetector\|LSTRModelType\|lstr_detector" --include="*.py" src/ config.py main.py
grep -r "HybridNetsClient\|HYBRIDNETS_CLIENT_AVAILABLE" --include="*.py" src/ config.py main.py
grep -r "_supercombo_client\|_detect_with_supercombo" --include="*.py" src/ config.py main.py
grep -r "signDetector\|SignDetector\|threadSignDetection" --include="*.py" src/ config.py main.py
grep -r "SIGN_SERVER_URL\|sign_server_url\|use_legacy_remote_sign_detection" --include="*.py" src/ config.py
grep -r "ai_analysis\|hybrid_fusion" --include="*.py" config.py
grep -rn "lstrAvailable\|lstrModels\|hybridnetsServer\|supercomboServer" --include="*.ts" src/dashboard/
```

### Tests automatizados:
```bash
python -m pytest tests/
```
Verificar que estos tests siguen pasando:
- `test_ai_local_lane_side_mapping.py`
- `test_stanley_controller.py`
- `test_stanley_physical_units.py`
- `test_thread_lane_observer.py`
- `test_pipeline_integration.py`

---

## 6. RESUMEN ESTIMADO DE CODIGO A ELIMINAR

| Categoria | LOC estimadas |
|-----------|--------------|
| `lstrDetector.py` (archivo completo) | ~554 |
| `signDetector.py` (archivo completo) | ~186 |
| `threadSignDetection.py` (archivo completo) | ~734 |
| `aiserver/` (directorio completo) | ~5,000+ |
| `threadLineFollowing.py` (metodos + bloques legacy) | ~1,500+ |
| `processCamera.py` (bloques legacy) | ~30 |
| `config.py` (entradas legacy) | ~5 |
| Dashboard `line-following.component.html` | ~200+ |
| Dashboard `line-following.component.ts` | ~200+ |
| Dashboard `line-following.component.css` | ~80+ |
| **TOTAL ESTIMADO** | **~8,500+ lineas** |

---

## 7. LO QUE SE MANTIENE (NO TOCAR)

| Componente | Razon |
|------------|-------|
| `src/data/Semaphores/` | Se va a usar en el futuro |
| `src/data/TrafficCommunication/` | Se va a usar en el futuro |
| `SystemMode.LEGACY` en `systemMode.py` | Necesario para habilitar Semaphores/TrafficCom |
| Transiciones `dashboard_legacy_button` en `transitionTable.py` | Necesario para acceder a modo LEGACY |
| `'legacy'` en `state-switch.component.ts` | Boton de modo LEGACY en dashboard |
| `main.py` lineas 65-66 (imports) y lineas comentadas | Se van a descomentar cuando se use |
| `signActions.py` | Lo usan `maneuverManager.py` y `threadLocalPerception.py` |
| `src/hardware/mpc/` | MPC en desarrollo activo |
