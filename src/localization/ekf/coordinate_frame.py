# src/localization/ekf/coordinate_frame.py
#
# Conversión GPS WGS84 ↔ marco ENU local. El EKF7 trabaja en metros sobre
# un plano tangente local (ENU = East-North-Up); el módulo encapsula la
# transformación con `pyproj` para mantener la matemática del filtro
# limpia de dependencias geodésicas.
#
# Por qué no usar lat/lon crudo:
#   - Las distancias en grados son anisotrópicas (1° lon en BFMC ≈ 75 km;
#     1° lat ≈ 111 km), y la métrica del Kalman asume euclídea.
#   - Los Jacobianos del modelo de planta se complican: mejor proyectar
#     una sola vez al ENU y operar siempre en metros.
#
# Decisión: proyección Transverse Mercator centrada en el origen
# configurado (`config.LOCAL_FRAME_ORIGIN_LATLON`). En BFMC la pista
# entera entra en ~50×50 m, así que la distorsión TM es despreciable
# (<<1 mm). La pista de Cluj se calibra una sola vez al inicio.
#
# La clase `GpsToEnu` se construye con (lat0, lon0) y expone:
#   to_enu(lat, lon) -> (east_m, north_m)     (relativo al origen)
#   from_enu(east_m, north_m) -> (lat, lon)   (inverso, útil para logs)

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer


@dataclass(frozen=True)
class LatLon:
    """Punto WGS84 (latitud + longitud en grados)."""

    lat_deg: float
    lon_deg: float


class GpsToEnu:
    """Proyecta GPS WGS84 al plano ENU local centrado en `origin`.

    Construir UNA instancia al arranque del proceso y reusar — el
    `Transformer` interno de pyproj cachea la rejilla de proyección.

    Parámetros:
        origin: punto WGS84 que se mapea a (0, 0) en ENU.

    Notas:
        - El plano ENU asume eje X = Este, eje Y = Norte. El yaw del
          vehículo se mide CCW desde el eje X (estándar matemático).
        - La altura (Up) no se usa: el modelo planar del EKF7 ignora Z.
        - `Transformer.transform` con `always_xy=True` devuelve siempre
          (x, y) = (east, north), independiente del CRS de destino.
    """

    def __init__(self, origin: LatLon) -> None:
        self._origin = origin
        # Transverse Mercator centrado en el origen. lat_0/lon_0 son los
        # parámetros estándar de proj.4; k=1.0 (sin escala), units=m.
        # Todo punto del plano que produce esta proyección está en
        # metros desde el origen, con +x → este, +y → norte.
        local_crs = CRS.from_proj4(
            f"+proj=tmerc +lat_0={origin.lat_deg} +lon_0={origin.lon_deg} "
            "+k=1.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        wgs84 = CRS.from_epsg(4326)
        # always_xy: forzar orden (lon, lat) en input y (x, y) en output.
        # Sin esto, pyproj usa el orden definido por el CRS, que es
        # (lat, lon) para WGS84 — fácil de equivocarse.
        self._fwd = Transformer.from_crs(wgs84, local_crs, always_xy=True)
        self._inv = Transformer.from_crs(local_crs, wgs84, always_xy=True)

    @property
    def origin(self) -> LatLon:
        return self._origin

    def to_enu(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Convierte un GPS WGS84 a (east_m, north_m) relativo al origen."""
        east, north = self._fwd.transform(lon_deg, lat_deg)
        return float(east), float(north)

    def from_enu(self, east_m: float, north_m: float) -> tuple[float, float]:
        """Inversa: (east, north) → (lat, lon). Útil para serializar pose."""
        lon, lat = self._inv.transform(east_m, north_m)
        return float(lat), float(lon)
