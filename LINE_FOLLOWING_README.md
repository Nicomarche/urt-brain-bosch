# Sistema de Seguimiento de Líneas con OpenCV

## Descripción
Este sistema permite que el vehículo siga automáticamente las líneas de la pista cuando está en modo AUTO. Utiliza OpenCV para detectar líneas blancas y amarillas, y controla automáticamente el steering (dirección) y la velocidad del vehículo.

## Cómo Funcionar

### 1. Activar el Modo AUTO
Para activar el seguimiento de líneas, simplemente cambia el sistema a modo AUTO desde el dashboard. El sistema automáticamente:
- Habilita la cámara
- Activa el thread de seguimiento de líneas
- Comienza a enviar comandos de control al vehículo

### 2. Funcionamiento Automático
El sistema:
- **Detecta líneas**: Identifica líneas blancas y amarillas en la pista
- **Calcula la dirección**: Determina el ángulo de steering necesario
- **Ajusta la velocidad**: Reduce velocidad en curvas cerradas
- **Controla el vehículo**: Envía comandos continuos de steering y speed

## Parámetros Configurables

Puedes ajustar estos parámetros en [`threadLineFollowing.py`](src/hardware/camera/threads/threadLineFollowing.py):

### Velocidad
```python
self.base_speed = 0.2        # Velocidad base
self.max_speed = 0.35        # Velocidad máxima en rectas
self.min_speed = 0.15        # Velocidad mínima en curvas
```

### Steering (Dirección)
```python
self.max_steering = 25.0              # Ángulo máximo de giro (grados)
self.steering_sensitivity = 0.8       # Sensibilidad (0-1): más alto = más reactivo
```

### Región de Interés (ROI)
```python
self.roi_height_start = 0.5       # Inicio del área de detección (50% de altura)
self.roi_height_end = 0.9         # Fin del área de detección (90% de altura)
self.roi_width_margin = 0.15      # Margen lateral (15% cada lado = 70% centro)
```

### Detección de Color (HSV)

**Líneas Blancas:**
```python
self.white_lower = np.array([0, 0, 200])
self.white_upper = np.array([180, 30, 255])
```

**Líneas Amarillas:**
```python
self.yellow_lower = np.array([20, 100, 100])
self.yellow_upper = np.array([30, 255, 255])
```

## Modo Debug

Para activar la visualización en tiempo real, modifica en [`processCamera.py`](src/hardware/camera/processCamera.py):

```python
lineFollowingTh = threadLineFollowing(
    self.queuesList, self.logging, self.debugging, 
    show_debug=True  # Cambia a True para ver el debug
)
```

Esto mostrará una ventana con:
- ✅ Líneas detectadas (verde)
- 🔵 Región de interés (azul)
- 🔴 Línea central de referencia
- 🟣 Centro de líneas detectado
- 📊 Valores de steering y speed

## Ajustes Recomendados

### Si el carro va muy rápido:
```python
self.base_speed = 0.15
self.max_speed = 0.25
```

### Si el carro no gira suficiente:
```python
self.max_steering = 30.0
self.steering_sensitivity = 1.0
```

### Si el carro gira demasiado:
```python
self.steering_sensitivity = 0.5
self.max_steering = 20.0
```

### Si no detecta las líneas:
1. Ajusta los valores HSV según la iluminación
2. Aumenta el área de ROI:
```python
self.roi_height_start = 0.3
self.roi_height_end = 0.95
```

### Si detecta demasiadas cosas falsas (objetos del costado):
1. Aumenta el margen lateral para visión más estrecha:
```python
self.roi_width_margin = 0.20  # 60% de ancho central
```
2. Ajusta los umbrales HSV para ser más restrictivos

### Si no detecta líneas punteadas:
1. Ajusta los parámetros de Hough para segmentos cortos:
```python
# En la función process_frame() - ya configurado para líneas punteadas
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=30,        # Umbral bajo para segmentos pequeños
    minLineLength=20,    # Segmentos cortos de líneas punteadas
    maxLineGap=150       # Gap grande para conectar puntos
)
```

### Si detecta líneas continuas sólidas:
1. Para líneas más continuas, aumenta los valores:
```python
threshold=70,        # Aumenta para ser más estricto
minLineLength=50,    # Aumenta para líneas más largas
maxLineGap=80        # Reduce para líneas más continuas
```

## Características Adicionales

### Control Adaptativo de Velocidad
El sistema reduce automáticamente la velocidad en curvas cerradas:
- Ángulo > 15°: velocidad mínima
- Ángulo > 10°: velocidad media
- Ángulo < 10°: velocidad máxima

### Manejo de Pérdida de Línea
Si no detecta líneas por varios frames:
- Reduce la velocidad automáticamente
- Mantiene el último steering conocido brevemente
- Se detiene si pierde la línea por mucho tiempo

### Filtrado de Ruido
El sistema utiliza:
- Operaciones morfológicas para limpiar la máscara
- Gaussian blur para suavizar
- Detección de bordes Canny
- Transformada de Hough para líneas robustas

## Solución de Problemas

### El vehículo no responde
1. Verifica que estés en modo AUTO
2. Confirma que la cámara esté funcionando
3. Revisa los logs en la terminal

### Detección erráctica
1. Activa el modo debug para visualizar
2. Ajusta los valores HSV según la iluminación de tu pista
3. Modifica el ROI si las líneas están muy cerca o lejos

### Oscilación excesiva
1. Reduce `steering_sensitivity`
2. Aumenta el suavizado (valores de Gaussian blur)
3. Ajusta los parámetros de HoughLinesP

## Archivos Modificados

1. **[`systemMode.py`](src/statemachine/systemMode.py)**: Habilita cámara y line following en modo AUTO
2. **[`threadLineFollowing.py`](src/hardware/camera/threads/threadLineFollowing.py)**: Nuevo thread para seguimiento de líneas
3. **[`processCamera.py`](src/hardware/camera/processCamera.py)**: Integra el thread de line following

## Próximas Mejoras Posibles

- 🚦 Detección de semáforos y señales
- 🚗 Detección de obstáculos
- 📏 Control PID para steering más suave
- 🎯 Predicción de trayectoria
- 🔄 Filtro de Kalman para suavizado
- 📊 Telemetría y logging de datos
