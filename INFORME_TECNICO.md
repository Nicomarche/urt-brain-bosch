# Informe Técnico: Sistema de Percepción y Control Autónomo para Vehículo a Escala

## Bosch Future Mobility Challenge (BFMC) 2026

---

## 1. Introducción

El presente documento describe el diseño e implementación del sistema de percepción visual y control autónomo desarrollado para un vehículo a escala 1:10 en el marco de la Bosch Future Mobility Challenge 2026. El sistema opera sobre una Raspberry Pi 5 como unidad de procesamiento central, con descarga de inferencia de redes neuronales a un servidor remoto con GPU mediante protocolo WebSocket.

El vehículo debe recorrer una pista que simula un entorno urbano, siguiendo carriles delimitados por líneas blancas, respetando señales de tráfico, semáforos, y ajustando su velocidad según las condiciones del camino.

### 1.1 Dimensiones del Vehículo

| Parámetro | Valor |
|---|---|
| Largo total | 36.5 cm |
| Ancho total | 19.0 cm |
| Distancia entre ejes (wheelbase) | 27.5 cm |
| Voladizo delantero (eje frontal a parachoques) | 7.2 cm |
| Posición de cámara (delante del eje frontal) | 11.5 cm |

### 1.2 Dimensiones de la Pista

| Parámetro | Valor |
|---|---|
| Ancho de carril (entre bordes internos de líneas) | 35.0 cm |
| Ancho de marcas de carril | 2.0 cm |
| Radio de curva interior (al centro del carril) | 66.5 cm |
| Radio de curva exterior (al centro del carril) | 103.5 cm |

---

## 2. Arquitectura del Sistema

El sistema emplea una arquitectura multiproceso donde cada módulo funcional corre en un proceso independiente del sistema operativo, comunicándose mediante colas de mensajes clasificadas por prioridad. Esto permite que la captura de cámara, el procesamiento de imagen, la comunicación serial con el microcontrolador y la interfaz web operen de manera concurrente sin interferencia mutua.

Los procesos principales son:

- **Proceso de Cámara**: Captura de frames y alojamiento de los hilos de seguimiento de carril y detección de señales.
- **Proceso Serial**: Comunicación bidireccional UART con el microcontrolador Nucleo STM32 que controla los actuadores (motor DC y servomotor de dirección).
- **Proceso Dashboard**: Servidor web que expone una interfaz Angular para monitoreo y control manual.
- **Proceso Gateway**: Enrutador interno de mensajes entre todos los procesos.

### 2.1 Máquina de Estados

El sistema opera en cuatro modos:

| Modo | Cámara | Seguimiento de Carril | Detección de Señales | Actuadores |
|---|---|---|---|---|
| DEFAULT | Activa | Inactivo | Inactivo | Inactivos |
| AUTO | Activa | Activo | Activo | Autónomos |
| MANUAL | Activa | Inactivo | Inactivo | Control remoto |
| STOP | Activa | Inactivo | Inactivo | Frenado |

La transición entre modos se realiza desde el dashboard web y se propaga a todos los módulos mediante mensajes de cambio de estado.

---

## 3. Pipeline de Seguimiento de Carril por Visión Computacional

El módulo de seguimiento de carril implementa un pipeline completo de procesamiento de imagen basado en OpenCV. A continuación se describe cada etapa en detalle.

### 3.1 Pre-procesamiento: Normalización de Iluminación

La variabilidad de iluminación (sombras, reflejos, zonas brillantes) es uno de los principales desafíos en detección de líneas. Se implementaron tres técnicas complementarias:

#### 3.1.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

La ecualización de histograma clásica opera sobre la imagen completa, lo que puede sobre-amplificar el ruido en regiones homogéneas. CLAHE resuelve esto dividiendo la imagen en bloques (tiles) y ecualizando cada uno independientemente con un límite de amplificación de contraste (clip limit).

**Procedimiento:**

1. Se convierte la imagen de BGR al espacio de color CIELAB, que separa la luminancia (canal L) de la información cromática (canales a, b).
2. Se aplica CLAHE únicamente al canal L con los parámetros:
   - **Clip Limit = 2.0**: Limita la amplificación máxima del contraste. Valores más altos permiten mayor contraste pero amplifican más ruido.
   - **Tile Grid Size = 8×8**: Cada bloque de 8×8 píxeles se ecualiza de manera independiente, adaptándose a las condiciones de iluminación locales.
3. Se recompone la imagen CIELAB y se convierte de vuelta a BGR.

El resultado es una imagen con iluminación perceptualmente uniforme, donde las líneas blancas mantienen su contraste tanto en zonas sombreadas como en zonas brillantes.

#### 3.1.2 Detección Adaptativa de Blanco

En lugar de usar un umbral de brillo fijo (que fallaría si la iluminación general cambia), se calcula el umbral dinámicamente basándose en la estadística del frame actual:

1. Se convierte la imagen a escala de grises.
2. Se calcula el percentil 92 de los valores de brillo. Esto significa que solo el 8% más brillante de los píxeles se considerará "blanco".
3. Se aplica un umbral mínimo de seguridad de 180 (escala 0–255) para evitar falsos positivos cuando toda la imagen es oscura.
4. Se genera una máscara binaria donde los píxeles por encima del umbral calculado son blancos.

Este enfoque se auto-ajusta a las condiciones de luz: en una escena oscura el percentil 92 será más bajo pero nunca inferior a 180, y en una escena brillante será naturalmente más alto.

#### 3.1.3 Fallback por Gradiente (Detección de Bordes)

Cuando la detección por color encuentra menos del 1% de píxeles en la imagen, se activa automáticamente un método de respaldo basado en detección de bordes. Este método es independiente del color y la iluminación, ya que detecta cambios bruscos de intensidad:

1. Se aplican los filtros de Sobel en las direcciones horizontal y vertical para obtener las derivadas parciales de la intensidad.
2. Se calcula la magnitud del gradiente: \( G = \sqrt{G_x^2 + G_y^2} \), donde \( G_x \) y \( G_y \) son las respuestas de Sobel en cada dirección.
3. Se normaliza la magnitud al rango [0, 255].
4. Se aplica un umbral basado en el percentil 85 de la magnitud del gradiente para retener solo los bordes más prominentes.

### 3.2 Segmentación por Color (Espacio HSV)

Tras el pre-procesamiento, se convierte la imagen al espacio HSV (Hue, Saturation, Value) que desacopla la información de color de la intensidad luminosa, facilitando la segmentación por color:

- **Detección de blanco**: H ∈ [81, 180], S ∈ [0, 98], V ∈ [200, 255]. El rango de saturación bajo y valor alto captura superficies blancas independientemente de su tono.
- **Detección de amarillo**: H ∈ [173, 86] (rango circular), S ∈ [100, 255], V ∈ [100, 255]. La saturación alta distingue el amarillo del blanco.

Se genera una máscara combinada mediante la operación OR bit a bit de las máscaras blanca, amarilla y adaptativa. Si el fallback por gradiente se activó, su máscara también se agrega con OR.

### 3.3 Región de Interés (ROI)

Se define una región de interés trapezoidal que descarta la porción superior de la imagen (cielo, edificios) y los márgenes laterales excesivos:

- **Inicio vertical**: 35% desde el borde superior (se descarta el primer tercio de la imagen).
- **Fin vertical**: 100% (borde inferior).
- **Márgenes laterales superiores**: 35% cada lado (visión estrecha en la lejanía).
- **Márgenes laterales inferiores**: 15% cada lado (visión más amplia cerca del auto).

La forma trapezoidal simula la perspectiva natural de la cámara, donde las líneas convergen hacia un punto de fuga en la parte superior.

### 3.4 Umbralización Binaria con Reintento

Se aplica un umbral binario global sobre la imagen en escala de grises dentro de la ROI:

1. **Primer intento**: Umbral de 165. Los píxeles con intensidad superior a 165 se marcan como blancos (potenciales líneas de carril).
2. Si no se detectan líneas suficientes con el primer umbral, se ejecuta un **segundo intento** con umbral de 90, que es más permisivo y captura líneas en condiciones de baja iluminación.

### 3.5 Filtrado de Ruido (Median Blur)

Se aplica un filtro de mediana con kernel 3×3 sobre la imagen binarizada. El filtro de mediana es particularmente efectivo para eliminar ruido tipo sal y pimienta (píxeles aislados blancos o negros) sin difuminar los bordes de las líneas, a diferencia del filtro gaussiano que suaviza todo uniformemente.

### 3.6 Detección de Bordes (Canny)

El detector de bordes Canny se aplica sobre la imagen filtrada con los umbrales:
- **Umbral bajo**: 100
- **Umbral alto**: 150

El algoritmo de Canny opera en tres etapas internas:
1. Calcula el gradiente de intensidad en cada píxel.
2. Aplica supresión de no-máximos para adelgazar los bordes a un solo píxel de ancho.
3. Aplica histéresis con los dos umbrales: los bordes con gradiente > 150 se aceptan siempre; los bordes con gradiente entre 100 y 150 se aceptan solo si están conectados a un borde fuerte.

### 3.7 Transformada de Hough Probabilística

Sobre la imagen de bordes Canny se aplica la transformada de Hough probabilística para detectar segmentos de línea recta:

| Parámetro | Valor | Descripción |
|---|---|---|
| ρ (rho) | 1 píxel | Resolución de distancia en el espacio de Hough |
| θ (theta) | π/180 rad | Resolución angular (1 grado) |
| Threshold | 50 votos | Mínimo de votos para considerar una línea |
| MinLineLength | 50 píxeles | Longitud mínima de un segmento detectado |
| MaxLineGap | 150 píxeles | Máxima separación entre segmentos para unirlos |

El MaxLineGap de 150 píxeles es deliberadamente alto para conectar líneas punteadas (dashed lines) que son comunes en las pistas BFMC.

### 3.8 Clasificación de Líneas por Pendiente

Cada segmento detectado por Hough se clasifica como línea izquierda o derecha según su pendiente:

1. Se calcula el ángulo de cada segmento: \( \theta = \arctan\left(\frac{y_2 - y_1}{x_2 - x_1}\right) \)
2. Se descartan las líneas con ángulo menor a 30° respecto a la horizontal (líneas casi horizontales que no son bordes de carril).
3. Las líneas con pendiente negativa (suben hacia la izquierda) se clasifican como **línea izquierda**.
4. Las líneas con pendiente positiva (suben hacia la derecha) se clasifican como **línea derecha**.

### 3.9 Fusión y Promediado de Líneas

Dentro de cada grupo (izquierda/derecha), las líneas cercanas se fusionan para reducir redundancia:

1. **Fusión**: Si el extremo inicial de un segmento está a menos de 175 píxeles del extremo final de otro, se unen en un solo segmento extendido.
2. **Promediado**: Todas las líneas restantes del grupo se promedian componente a componente (x₁, y₁, x₂, y₂) para obtener una única línea representativa por lado.

### 3.10 Cálculo del Error Lateral

Con las dos líneas promediadas (izquierda y derecha), se calcula el error lateral del vehículo:

1. Se extiende cada línea hasta el borde inferior de la imagen (la posición más cercana al auto).
2. Se calcula el punto medio entre las dos intersecciones con el borde inferior.
3. El **error** es la diferencia horizontal entre este punto medio y el centro de la imagen: \( e = x_{\text{midpoint}} - x_{\text{center}} \)

Un error positivo indica que el vehículo está desplazado hacia la derecha; un error negativo indica desplazamiento hacia la izquierda.

#### Caso de una sola línea visible

Cuando solo se detecta una línea (situación común al entrar en curvas), se utiliza un método alternativo basado en la pendiente de la línea visible:

- **Pendiente > 50°** respecto a la horizontal → línea casi vertical → corrección de 3°
- **Pendiente entre 40° y 50°** → curva moderada → corrección de 11°
- **Pendiente < 40°** → curva pronunciada → corrección de 22°

La dirección de la corrección se invierte según si la línea visible es la izquierda o la derecha.

---

## 4. Filtro de Rechazo de Spikes (Noise Rejection)

Los reflejos, el brillo solar directo y las superficies metálicas generan detecciones espurias que, sin filtrado, producirían saltos bruscos en el steering (spikes). Se implementó un filtro multi-criterio que evalúa cada frame antes de aceptar sus resultados:

### 4.1 Criterios de Rechazo

Un frame se marca como **ruidoso** y se rechaza si cumple cualquiera de estas condiciones:

| Criterio | Umbral | Justificación |
|---|---|---|
| Exceso de líneas Hough | > 40 líneas | Los reflejos generan cientos de bordes falsos que producen una explosión de líneas en el espacio de Hough. En condiciones normales, el carril genera entre 2 y 15 líneas. |
| Salto de error entre frames | > 80 píxeles | En operación normal, la posición del carril cambia gradualmente (< 20px/frame a velocidad de crucero). Un salto de 80px indica una detección corrupta. |
| Salto de steering entre frames | > 15 grados | Un cambio de 15° entre frames consecutivos a ~5 FPS implicaría una maniobra físicamente imposible. |
| Pérdida súbita de ambas líneas | 2 líneas → 0 líneas | Si en el frame anterior se veían ambas líneas y en el actual ninguna, es más probable un fallo de detección que una pérdida real. |

### 4.2 Comportamiento del Filtro

- Cuando un frame es rechazado, se mantienen el último error y steering aceptados. El vehículo continúa con la última dirección conocida.
- Se permite un máximo de **3 frames consecutivos rechazados**. Si se alcanza este límite, el siguiente frame se acepta incondicionalmente para evitar que el sistema quede "congelado" ante un cambio real del entorno.
- **Excepción**: Durante las transiciones de curva (estados ENTERING y EXITING), el filtro de salto de steering se desactiva, ya que en esos momentos los cambios bruscos son legítimos.
- Cada vez que un frame es aceptado, se registra como la nueva referencia para comparaciones futuras.

---

## 5. Controlador PID

El ángulo de dirección se calcula mediante un controlador PID (Proporcional-Integral-Derivativo) que recibe como entrada el error lateral normalizado y produce como salida el ángulo de steering en grados.

### 5.1 Formulación Matemática

\[
u(t) = K_p \cdot e(t) + K_i \int_0^t e(\tau) \, d\tau + K_d \cdot \frac{de(t)}{dt}
\]

Donde:
- \( e(t) \) es el error lateral normalizado en el rango [-1, 1]
- \( u(t) \) es el ángulo de steering en grados, limitado a ±25°

### 5.2 Parámetros

| Parámetro | Valor | Función |
|---|---|---|
| Kp (Proporcional) | 25.0 | Corrección inmediata proporcional al error actual. Un error normalizado de 1.0 genera 25° de steering. |
| Ki (Integral) | 1.0 | Acumula el error a lo largo del tiempo para corregir desviaciones persistentes que el término proporcional no puede eliminar (por ejemplo, un sesgo constante de la cámara). |
| Kd (Derivativo) | 4.0 | Anticipa la trayectoria futura observando la tasa de cambio del error. Amortigua las oscilaciones causadas por un Kp agresivo. |

### 5.3 Mejoras sobre el PID Clásico

#### Zona Muerta (Dead Zone)
Se define una zona muerta de 50 píxeles: si el error absoluto es menor a este valor, la salida del PID es cero y se resetea el término integral. Esto previene micro-oscilaciones cuando el vehículo está correctamente centrado en el carril, donde pequeñas perturbaciones de la cámara generarían correcciones innecesarias.

#### Anti-Windup del Integral
El término integral se resetea a cero cada 10 iteraciones. Sin esta medida, el integral acumularía error indefinidamente durante curvas prolongadas, y al salir de la curva generaría un overshoot significativo antes de descargarse. Este mecanismo de "reset periódico" es una forma simple pero efectiva de anti-windup.

#### Delta-T Real
En lugar de asumir un intervalo de tiempo fijo entre frames, se mide el delta-t real entre llamadas consecutivas al controlador usando timestamps del sistema. Esto asegura que el término integral acumule correctamente y que el término derivativo calcule la tasa de cambio real, incluso si la tasa de frames varía (por ejemplo, si un frame tarda más en procesarse).

### 5.4 Feed-Forward por Curvatura (Modelo de Ackermann)

Cuando el sistema detecta una curva mediante la estimación de curvatura del carril, se añade un componente feed-forward al PID que pre-calcula el ángulo de dirección geométricamente necesario:

\[
\delta_{\text{ff}} = \arctan\left(\frac{L}{R}\right)
\]

Donde:
- \( L = 0.265 \) m es la distancia entre ejes del vehículo
- \( R \) es el radio de curvatura estimado en metros

La salida final combina ambos componentes:

\[
\delta = w_{\text{ff}} \cdot \delta_{\text{ff}} + (1 - w_{\text{ff}}) \cdot \delta_{\text{PID}}
\]

Con \( w_{\text{ff}} = 0.6 \), lo que asigna 60% al feed-forward geométrico y 40% a las correcciones finas del PID. En línea recta (\( \delta_{\text{ff}} \approx 0 \)), el PID opera en solitario.

### 5.5 Control Adaptativo de Velocidad

La velocidad del vehículo se ajusta automáticamente en función del ángulo de steering actual:

| Ángulo de Steering | Velocidad | Justificación |
|---|---|---|
| < 10° | Máxima (10 unidades) | Recta: el vehículo puede avanzar a velocidad de crucero. |
| 10° – 15° | Media (interpolación lineal) | Curva moderada: se reduce gradualmente la velocidad. |
| > 15° | Mínima (5 unidades) | Curva cerrada: se necesita velocidad baja para mantener la trayectoria. |

Adicionalmente, se implementa una **rampa de aceleración** que limita el incremento de velocidad a 0.5 unidades por frame, evitando cambios bruscos que podrían desestabilizar el vehículo.

---

## 6. Máquina de Estados de Curvas

La navegación en curvas presenta un desafío particular: al entrar en una curva, una de las dos líneas de carril desaparece del campo de visión de la cámara. Se implementó una máquina de estados finitos que modela las fases de una curva:

### 6.1 Estados y Transiciones

```
STRAIGHT ──────► ENTERING ──────► IN_CURVE ──────► EXITING ──────► STRAIGHT
(2 líneas)      (transición)      (1 línea)       (transición)     (2 líneas)
```

| Transición | Condición |
|---|---|
| STRAIGHT → ENTERING | Se detecta solo 1 línea durante ≥1 frame consecutivo, O el punto de fuga se desvía más del 20% del centro de la imagen durante ≥3 frames. |
| ENTERING → IN_CURVE | Se confirma 1 sola línea durante ≥2 frames consecutivos. |
| IN_CURVE → EXITING | Se vuelven a detectar 2 líneas durante ≥3 frames. |
| EXITING → STRAIGHT | Transición automática cuando la curvatura medida desciende. |

### 6.2 Estimación del Punto de Fuga

Cuando ambas líneas son visibles (estado STRAIGHT), se calcula el punto de fuga (vanishing point) como la intersección de las dos líneas extrapoladas. La posición horizontal del punto de fuga relativa al centro de la imagen indica la dirección de una curva inminente:

- **VP desplazado a la derecha** → la pista gira a la derecha
- **VP desplazado a la izquierda** → la pista gira a la izquierda

Esto permite detectar una curva **antes** de que desaparezca una de las líneas.

### 6.3 Recuperación de Curva (Maniobra de Reversa)

Si el vehículo queda saturado en máximo steering (≥ 90% del límite de 25°) durante más de 8 frames consecutivos y el error no está disminuyendo, se determina que el auto no puede completar la curva desde su posición actual. Se ejecuta una maniobra de recuperación en 5 fases:

1. **STOPPING** (100 ms): Frenado completo.
2. **PRE_TURNING** (600 ms): Con el auto detenido, se giran las ruedas al ángulo de reversa calculado. Se espera a que el servomotor posicione las ruedas físicamente.
3. **REVERSING** (300–2500 ms, variable): El auto retrocede con un ángulo de dirección fijo opuesto a la curva. La duración es proporcional a la magnitud del error: un error pequeño genera una reversa corta, un error grande genera una reversa más prolongada.
4. **REALIGNING** (600 ms): Se detiene y se giran las ruedas hacia la dirección de la curva (máximo steering) para preparar el reingreso.
5. **RESUMING** (100 ms): Se resetea el PID y se reanuda la marcha hacia adelante.

**Guardas de seguridad**: La recuperación solo se activa durante los estados ENTERING o IN_CURVE. Si el auto está corrigiendo exitosamente (el error está disminuyendo), la recuperación no se activa aunque el steering esté al máximo.

---

## 7. Detección de Señales de Tráfico

El sistema detecta señales de tráfico mediante una arquitectura de inferencia remota: la Raspberry Pi envía frames a un servidor con GPU, que ejecuta un modelo de detección de objetos y devuelve las señales encontradas.

### 7.1 Modelo de Detección

Se utiliza un modelo **YOLOv8** (You Only Look Once, versión 8) entrenado específicamente para señales de tráfico. YOLOv8 es un detector de objetos de una sola pasada (single-shot) que divide la imagen en una grilla y predice simultáneamente las coordenadas del bounding box, la clase y la confianza para cada celda de la grilla.

Características del modelo:
- **Tamaño de entrada**: 320×320 píxeles (configurable hasta 640 para mayor precisión)
- **Peso del modelo**: 22 MB
- **Clases detectadas**: 25+ categorías incluyendo stop, semáforos (rojo/amarillo/verde), límites de velocidad (20/30 km/h), cruces peatonales, estacionamiento, entrada/salida de autopista, entre otros.

### 7.2 Protocolo de Comunicación

La comunicación entre la Raspberry Pi y el servidor se realiza mediante WebSocket, un protocolo de comunicación full-duplex sobre TCP que mantiene la conexión abierta para minimizar la latencia:

1. **RPi → Servidor**: Bytes JPEG crudos del frame de la cámara. La RPi no decodifica ni re-codifica la imagen; transmite los bytes JPEG directamente desde el buffer de la cámara, ahorrando ~15 ms de CPU por frame.
2. **Servidor → RPi**: Respuesta JSON compacta con las detecciones:

```json
{
  "d": [
    {"s": "stop", "c": 0.95, "b": [0.12, 0.30, 0.45, 0.62]}
  ],
  "t": 12.3,
  "f": 42
}
```

Donde `s` es el nombre de la señal, `c` es la confianza (0–1), `b` son las coordenadas normalizadas del bounding box [ymin, xmin, ymax, xmax], `t` es el tiempo de inferencia en milisegundos, y `f` es el identificador de frame.

### 7.3 Reconexión y Tolerancia a Fallos

El cliente WebSocket opera en un hilo daemon con un event loop asíncrono propio. Si la conexión se pierde, reintenta automáticamente cada 3 segundos. Los frames se almacenan en una cola de tamaño 1 (solo el frame más reciente), de modo que si el servidor está ocupado, no se acumulan frames obsoletos.

Si se producen 3 timeouts consecutivos (sin respuesta en 2 segundos), el cliente cierra la conexión y la reabre para forzar una reconexión limpia.

### 7.4 Sistema de Acciones Vehiculares

Cuando se detecta una señal con confianza suficiente y el vehículo está en modo autónomo, se ejecutan acciones predefinidas:

| Señal Detectada | Acción del Vehículo |
|---|---|
| Stop / No Entry / Semáforo Rojo | Frenado completo durante 3 segundos, luego reanuda velocidad previa. |
| Cruce Peatonal | Reduce velocidad al mínimo durante 3 segundos. |
| Semáforo Amarillo | Reduce velocidad al mínimo (sin detenerse). |
| Semáforo Verde | Reanuda velocidad normal. |
| Velocidad 20 / Velocidad 30 | Modifica la velocidad base del vehículo. |
| Entrada de Autopista | Incrementa velocidad a 7 unidades y activa modo autopista (velocidades más altas en el seguimiento de carril). |
| Salida de Autopista | Reduce velocidad a 5 unidades y desactiva modo autopista. |
| Estacionamiento | Detiene el vehículo indefinidamente. |

### 7.5 Filtros de Seguridad

Se implementan tres filtros para evitar acciones incorrectas:

#### 7.5.1 Cooldown por Grupo de Acción

Las señales se agrupan por tipo de acción (por ejemplo, "stop", "no entry" y "semáforo rojo" pertenecen al grupo "stop"). Después de ejecutar una acción, se ignoran todas las señales del mismo grupo durante 15 segundos. Esto evita que el vehículo frene múltiples veces al pasar junto a una misma señal de stop que aparece en varios frames consecutivos.

#### 7.5.2 Área Mínima de Bounding Box

Se calcula el área del bounding box como fracción del área total de la imagen. Si el área es menor al 1% (la señal está lejos), la detección se publica para telemetría pero **no se ejecuta ninguna acción**. Esto evita frenados prematuros cuando el vehículo detecta una señal que está a varios metros de distancia.

#### 7.5.3 Coordinación con Seguimiento de Carril

Las acciones de señales y el seguimiento de carril comparten un mecanismo de exclusión mutua basado en eventos. Cuando una acción de señal se activa (por ejemplo, frenado por stop), se señaliza un evento que bloquea al hilo de seguimiento de carril de enviar comandos de motor. Cuando la acción termina, el evento se libera y el seguimiento de carril retoma el control. Esto previene conflictos donde el seguimiento de carril intentaría acelerar mientras la acción de señal intenta frenar.

---

## 8. Modos de Detección Alternativos

Además del pipeline OpenCV descrito en la Sección 3, el sistema soporta cuatro modos de detección adicionales, seleccionables en tiempo real desde el dashboard:

### 8.1 LSTR (Lane Shape Prediction with Transformers)

Modelo de deep learning basado en la arquitectura Transformer que predice la forma geométrica completa de los carriles de manera end-to-end. A diferencia de los métodos clásicos que detectan píxeles individuales, LSTR predice directamente los parámetros de una ecuación paramétrica que describe la curva del carril:

\[
x(y) = \frac{k_2}{(y - f_2)^2} + \frac{m_2}{y - f_2} + n_1 + b_2 \cdot y - b_3
\]

El modelo se ejecuta localmente en la Raspberry Pi usando ONNX Runtime. Es más robusto a cambios de iluminación porque aprende características visuales en lugar de depender de umbrales de color.

### 8.2 Modo Híbrido (OpenCV + LSTR)

Ejecuta ambos detectores en paralelo y combina sus resultados con pesos configurables (40% OpenCV, 60% LSTR). Cuando ambos métodos coinciden en la dirección de corrección (diferencia < 5°), se aplica un multiplicador de confianza de 1.2×. Si un método falla, se usa el otro con confianza reducida.

### 8.3 HybridNets (Servidor Remoto)

Red neuronal multi-tarea que realiza simultáneamente segmentación de área transitable, detección de líneas de carril y detección de objetos. La inferencia se ejecuta en un servidor con GPU (PyTorch) y los resultados se transmiten por WebSocket. El servidor calcula directamente el ángulo de steering a partir de la segmentación.

### 8.4 Supercombo (Modelo de openpilot)

Modelo recurrente desarrollado por comma.ai para conducción autónoma real. Procesa 2 frames consecutivos en formato YUV con un estado interno GRU de 512 dimensiones que mantiene memoria temporal entre frames. Predice 4 líneas de carril con 33 puntos tridimensionales cada una, además de 5 trayectorias planeadas.

---

## 9. Optimización del Uso de CPU

La Raspberry Pi 5, a pesar de contar con un procesador de 4 núcleos ARM Cortex-A76, tiene recursos limitados cuando debe ejecutar simultáneamente captura de cámara, procesamiento de imagen, comunicación serial, servidor web y transmisión de video. Se realizó un trabajo sistemático de optimización que redujo significativamente la carga de CPU y permitió que el sistema opere de manera estable en tiempo real.

### 9.1 Afinidad de CPU (CPU Pinning)

Al inicio de la aplicación, se fija la afinidad de CPU del proceso principal a todos los núcleos disponibles mediante `psutil.Process().cpu_affinity()`. Esto garantiza que el scheduler del sistema operativo no restrinja los procesos hijos a un subconjunto de núcleos, y que la carga se distribuya uniformemente entre los 4 cores del BCM2712.

### 9.2 Eliminación de Procesos No Utilizados

El proyecto base de BFMC incluía dos procesos pesados que no son necesarios para la conducción autónoma en el entorno de pruebas:

- **processSemaphores**: Escuchaba por UDP paquetes de un servidor de semáforos. No utilizado en nuestro entorno.
- **processTrafficCommunication**: Mantenía conexión TCP con un servidor de tráfico para localización GPS. No utilizado.

Cada proceso implicaba un fork del intérprete de Python (~30 MB de RAM) más sus hilos internos de polling. La eliminación de ambos liberó aproximadamente 60 MB de RAM y dos procesos completos con sus hilos asociados.

### 9.3 Calibración de Tasas de Polling por Hilo

El framework base usaba un valor de `pause=0.001` (1000 Hz) para todos los hilos, lo que significa que cada hilo ejecutaba su ciclo de trabajo 1000 veces por segundo, consumiendo CPU en esperas activas innecesarias. Se calibró cada hilo con una tasa de polling acorde a su función real:

| Hilo | Tasa Original | Tasa Optimizada | Justificación |
|---|---|---|---|
| threadCamera | 1000 Hz | 10 Hz (100 ms) | La cámara produce ~5 FPS; muestrear más rápido es desperdicio. |
| threadLineFollowing | 1000 Hz | 20 Hz (50 ms) | Depende de la cámara a ~5 FPS; 20 Hz da margen sin desperdiciar. |
| threadSignDetection | 1000 Hz | 20 Hz (50 ms) | Rate limited a ~3 FPS de detección por el servidor remoto. |
| threadRead (serial) | 1000 Hz | 50 Hz (20 ms) | La telemetría serial del Nucleo no cambia más rápido que esto. |
| threadWrite (serial) | 1000 Hz | 100 Hz (10 ms) | Necesita respuesta rápida para comandos de motor, pero no 1 kHz. |
| threadGateway | 1000 Hz | 50 Hz (20 ms) | El enrutador de mensajes no necesita latencia sub-milisegundo. |
| Dashboard (mensajes) | Continuo | 7 Hz (150 ms) | La interfaz web no requiere actualización más rápida que esto. |
| Dashboard (hardware) | Continuo | 0.33 Hz (3 s) | Métricas de CPU/RAM/temperatura cambian lentamente. |

El mecanismo de espera usa `Event.wait(timeout)` en lugar de `time.sleep()`, lo que permite despertar inmediatamente ante una señal de parada sin esperar a que expire el timeout completo.

### 9.4 Eliminación de Codificación de Video Redundante

La cámara captura dos resoluciones simultáneamente:
- **mainCamera**: 2048×1080 (resolución completa)
- **serialCamera**: 640×384 (resolución reducida, usada por todos los consumidores)

En el proyecto base, ambos frames se codificaban a JPEG y se enviaban por las colas de mensajes. Se identificó que **ningún suscriptor consume mainCamera**: ni el line following, ni el sign detection, ni el dashboard la utilizan. Se eliminó completamente su codificación JPEG y transmisión, ahorrando la compresión de un frame de 2048×1080 en cada ciclo (~20-30 ms de CPU por frame).

### 9.5 Desactivación Condicional del Stream de Video al Dashboard

Se agregó un parámetro de configuración `STREAM_CAMERA_TO_DASHBOARD` (default: `False`). Cuando está desactivado:

1. El dashboard no se suscribe al mensaje `serialCamera`, por lo que la imagen nunca viaja por la cola al proceso web.
2. Se evita la serialización JSON + emisión WebSocket de ~50 KB de base64 por frame.
3. Esto ahorra tanto CPU (codificación base64, serialización) como ancho de banda de red.

Cuando se necesita ver el video en el dashboard (por ejemplo, durante calibración), se puede activar cambiando el flag a `True`.

### 9.6 Optimización del Buffer de Cámara

#### PiCamera (CSI)
Se configuró `buffer_count=1` y `queue=False` en la configuración de picamera2. Con `buffer_count=1`, la cámara mantiene un solo buffer en memoria en lugar del default de 4, reduciendo el uso de RAM en ~12 MB (3 buffers × 2048×1080×3 bytes). Con `queue=False`, cada llamada a `capture_array()` devuelve el frame más reciente en lugar de encolar frames, eliminando la latencia acumulada.

#### USB Camera (V4L2)
Se implementó una estrategia de **grab/retrieve separado** con un hilo lector dedicado:

1. Un hilo daemon ejecuta `grab()` en bucle cerrado. Esta operación es instantánea: simplemente extrae el frame más reciente del buffer V4L2 del kernel sin decodificarlo.
2. Solo cuando el consumidor ha procesado el frame anterior, el hilo llama a `retrieve()` (decodificación MJPEG → NumPy, operación costosa ~5-10 ms).
3. Esto garantiza que el frame decodificado es siempre el más reciente capturado, no uno obsoleto atrapado en un buffer. Sin esta optimización, V4L2 acumula frames en su buffer interno y entrega frames con hasta 200 ms de antigüedad.

Se configuró además `CAP_PROP_BUFFERSIZE=1` y formato `MJPG` para minimizar la latencia del driver.

### 9.7 Bypass de Decodificación en Detección de Señales

La cámara produce frames codificados en JPEG que se transmiten como base64 por las colas de mensajes. Normalmente, para enviar un frame al servidor de detección se requiere:

1. `base64.b64decode()` → bytes JPEG crudos
2. `cv2.imdecode()` → array NumPy BGR
3. `cv2.imencode()` → bytes JPEG crudos (para enviar por WebSocket)

Se identificó que los pasos 2 y 3 son redundantes: la imagen se decodifica a NumPy solo para volver a codificarse a JPEG. Se eliminó este ciclo y se envían los bytes JPEG crudos directamente al servidor:

```
base64.b64decode() → bytes JPEG → WebSocket (directo, sin decode/encode)
```

Esto ahorra ~10-15 ms de CPU por frame, que a 5 FPS representa un ahorro de 50-75 ms por segundo de procesamiento.

### 9.8 Envío No Bloqueante al Servidor de Señales

El cliente WebSocket de detección de señales opera en modo no bloqueante (`block=False`): envía el frame al servidor y continúa inmediatamente sin esperar la respuesta. Los resultados se recogen en la siguiente iteración del hilo. Esto desacopla la latencia de red/inferencia del servidor (~50-100 ms) del ciclo de captura de la cámara.

La cola interna del cliente tiene tamaño máximo 1, lo que garantiza que siempre se procesa solo el frame más reciente, descartando automáticamente los obsoletos si el servidor es más lento que la cámara.

### 9.9 Modo "lastOnly" en Suscriptores de Mensajes

Todos los suscriptores de mensajes operan en modo `"lastOnly"`: solo retienen el mensaje más reciente de cada tipo, descartando silenciosamente los anteriores. Esto previene:

- **Acumulación de colas**: Sin esta política, un consumidor lento acumularía miles de mensajes en la cola, consumiendo memoria creciente.
- **Procesamiento de datos obsoletos**: Procesar un frame de cámara de hace 2 segundos es peor que no procesarlo. "lastOnly" garantiza que siempre se trabaja con la información más actual.

### 9.10 Generación Condicional de Imágenes de Debug

Las imágenes de debug (visualizaciones de bordes, máscaras, overlays) solo se generan cuando la depuración visual está efectivamente activada. Se implementó una propiedad `_needs_debug` que evalúa si hay algún consumidor activo (ventana local abierta o stream de dashboard habilitado). Cuando el debug está desactivado:

- Se omiten todas las operaciones de `copy()`, `cv2.putText()`, `cv2.circle()`, `cv2.line()` sobre frames de debug.
- Se omite la codificación JPEG y base64 del stream de debug.
- Se omite el envío del mensaje `LineFollowingDebug` al dashboard.

En total, esto evita hasta 6 copias de frame y decenas de operaciones de dibujo por ciclo de procesamiento.

### 9.11 Descarga de Inferencia a GPU Remota

La optimización de mayor impacto es arquitectural: toda la inferencia de redes neuronales (HybridNets, Supercombo, YOLOv8) se ejecuta en un servidor remoto con GPU en lugar de en la Raspberry Pi. Los modelos de deep learning son extremadamente intensivos en cómputo:

| Modelo | Tiempo en CPU (RPi) | Tiempo en GPU (servidor) |
|---|---|---|
| YOLOv8 (sign detection) | ~800-1200 ms | ~10-20 ms |
| HybridNets (segmentación) | No viable (~5+ s) | ~30-80 ms |
| Supercombo (openpilot) | ~200-400 ms | ~15-40 ms |

Al descargar estos modelos al servidor, la Raspberry Pi solo necesita codificar el frame a JPEG (~5 ms) y enviarlo por WebSocket (~5 ms), reduciendo la carga de inferencia de ~1000 ms a ~10 ms por frame en el lado de la RPi.

### 9.12 Calidad JPEG Reducida

La codificación JPEG de los frames de cámara se realiza con calidad 70 (en una escala de 1-100) en lugar del default de 95. La reducción de calidad de 95 a 70 reduce el tamaño del archivo comprimido aproximadamente a la mitad (~50 KB vs ~100 KB para un frame 640×384), lo que reduce proporcionalmente:

- El tiempo de codificación JPEG en la RPi.
- El ancho de banda consumido en las colas internas y el WebSocket.
- El tiempo de decodificación en el servidor.

La pérdida de calidad visual es mínima para los algoritmos de detección, que trabajan con versiones re-escaladas y procesadas de la imagen.

### 9.13 Resumen de Impacto

| Optimización | Ahorro Estimado |
|---|---|
| Eliminar procesos no usados | ~2 procesos, ~60 MB RAM |
| Calibrar tasas de polling | ~30-40% menos wakeups de CPU |
| Eliminar codificación mainCamera | ~20-30 ms/frame |
| Desactivar stream al dashboard | ~15-20 ms/frame + ancho de banda |
| Bypass decode/encode en sign detection | ~10-15 ms/frame |
| Buffer de cámara optimizado | ~100-200 ms menos latencia |
| Debug condicional | ~5-10 ms/frame cuando desactivado |
| GPU offload | ~800-1200 ms/frame de inferencia liberados |

El efecto combinado de estas optimizaciones permitió que el sistema completo (cámara + procesamiento + serial + dashboard) opere de manera estable a ~5 FPS de procesamiento con un uso de CPU promedio del 40-60% en la Raspberry Pi 5, dejando margen suficiente para picos de carga y el sistema operativo.

---

## 10. Conclusiones

El sistema implementado integra múltiples técnicas de visión computacional, aprendizaje profundo y teoría de control para lograr un vehículo autónomo funcional a escala. Las contribuciones principales son:

1. **Pipeline de visión robusto**: La combinación de CLAHE, detección adaptativa y fallback por gradiente permite operar en condiciones de iluminación variable sin requerir calibración manual.

2. **Filtro de spikes multi-criterio**: Previene comportamiento errático ante reflejos y perturbaciones visuales, manteniendo la estabilidad del control.

3. **Control PID + Feed-Forward**: La combinación de control reactivo (PID) con predicción geométrica (Ackermann) logra un balance entre respuesta rápida y estabilidad.

4. **Máquina de estados de curvas**: La anticipación de curvas mediante el punto de fuga y la recuperación automática mediante reversa resuelven los escenarios más desafiantes de la pista.

5. **Arquitectura de inferencia distribuida**: La descarga de modelos pesados a un servidor GPU permite mantener la Raspberry Pi dedicada al control en tiempo real, mientras se aprovechan modelos de última generación para la percepción.

6. **Detección de señales con acciones autónomas**: El sistema de cooldown por grupo, área mínima y coordinación por eventos garantiza respuestas correctas y seguras ante señales de tráfico.

7. **Optimización integral de CPU**: El trabajo sistemático de calibración de polling, eliminación de codificaciones redundantes, bypass de ciclos decode/encode y generación condicional de debug permitió operar el sistema completo de manera estable en una plataforma embebida con recursos limitados.
