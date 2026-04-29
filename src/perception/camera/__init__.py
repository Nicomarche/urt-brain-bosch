# src/perception/camera/
#
# Captura de frames + calibración + transformación BEV (Bird's-Eye View).
#
#   threadCameraCapture.py   — solo `cv2.VideoCapture` + buffer compartido.
#                               Lo más liviano posible para que el frame
#                               rate no caiga; cero lógica de procesamiento.
#   bev_transform.py         — homografía cámara→plano de tierra usando
#                               intrinsics + altura del sensor.
#   calibration.py           — loader de archivos .yaml/.json en
#                               `calibration/` (intrinsics + distorsión).
#
# Hoy `src/hardware/camera/threads/threadCamera.py` cubre la captura;
# se desplaza acá cuando el pipeline nuevo absorbe responsabilidades.
