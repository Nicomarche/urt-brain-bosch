# `legacy/` — Componentes deprecated

Esta carpeta contiene código que se mantiene **solo para retrocompatibilidad** mientras
se completa la migración a la nueva GUI PyQt5 (ver `src/dashboard/gui/`).

## Contenido

### `dashboard-frontend/` (antes `src/dashboard/frontend/`)

Dashboard Angular 18 servido en `:4200` que se conecta al backend Flask+SocketIO de
`src/dashboard/processDashboard.py:5005`.

- **Estado:** deprecated. La nueva GUI PyQt5 (`src/dashboard/gui/`) cubre el 100% de
  esta funcionalidad y se conecta al mismo SocketIO.
- **Por qué se mantiene:** el servicio `services/angular-autostart/` puede seguir
  arrancando esta UI en el Jetson para usuarios que prefieran navegador, hasta que
  la GUI PyQt5 demuestre paridad completa en producción.
- **No agregar features nuevos acá.** Bug fixes solo si bloquean la operación del
  auto.

#### Si necesitás compilarlo:

```bash
cd legacy/dashboard-frontend
npm install
npm run build       # output en legacy/dashboard-frontend/dist/
npm start           # dev server en :4200
```

#### Cómo se conecta al backend:

El frontend lee la IP del backend desde `webSocket/web-socket.service.ts` (campo
hardcoded). El script `src/dashboard/components/ip_manger.py` reemplaza esa IP
automáticamente al arrancar `processDashboard.py`. **Si movés el path del frontend,
actualizá `ip_manger.py:31`.**

## Cómo eliminar definitivamente

Una vez que la GUI PyQt5 cubra todos los casos de uso en producción:

1. Desactivar el servicio: `sudo ./services/angular-autostart/uninstall.sh`
2. Eliminar `legacy/dashboard-frontend/` y `services/angular-autostart/`
3. Eliminar `src/dashboard/components/ip_manger.py` y la llamada `IpManager.replace_ip_in_file()` en `processDashboard.py`
