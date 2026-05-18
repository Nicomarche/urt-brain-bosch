# Legacy maps & track editor artifacts

Esta carpeta contiene archivos heredados que **NO se cargan en runtime**.
Se mueven aquí (plan E3) para liberar la raíz del repo y dejar trazabilidad.

## Contenido

| Archivo | Origen | Estado | Por qué está acá |
|---|---|---|---|
| `Track GraphML File.graphml` | Editor BFMC oficial | obsoleto | Reemplazado por `maps/jetson/lanelet2_map.osm` y `maps/sim/lanelet2_map.osm`. La única referencia residual es un docstring en [src/routing/visualizer.py:69](../../src/routing/visualizer.py:69) — *no carga el archivo*. |
| `Track GraphML Fileold.graphml` | Editor BFMC oficial | obsoleto | Versión anterior del anterior. |
| `Track Editor Save.json` | Editor BFMC oficial | obsoleto | Metadatos del editor (metersPerPixel, imgW, imgH). Reemplazado por `maps/{jetson,sim}/track_meta.json`. |
| `Track Editor SaveOld.json` | Editor BFMC oficial | obsoleto | Versión anterior. |
| `newComponent.py` | Skeleton Component generator | obsoleto | Plantilla legacy del template SocketIO de BFMC. 0 referencias activas en el código. |

## Si necesitás abrirlos

Los archivos del editor BFMC se abren con el Track Editor original (Java
swing). No tocar para producción — el repo ya migró a Lanelet2 OSM.

## Por qué no se eliminan

* Trazabilidad: alguien podría querer comparar la topología legacy contra
  el OSM nuevo.
* Tamaño: <1 MB total. No vale la pena el riesgo de borrar.

## Reglas

* **NO** importar nada desde `legacy/maps/` en código nuevo.
* **NO** referenciar estos archivos desde tests.
* Si encontrás una referencia activa (no docstring), abrir issue.
