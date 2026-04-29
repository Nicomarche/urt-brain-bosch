# src/perception/stopline/
#
# Detector de líneas de stop. Toma la máscara BEV (white-line mask) y
# devuelve una `StoplineObservation` con distancia + confianza.
#
#   stopline_detector.py  — proyecta el bbox de la línea blanca en
#                            coordenadas BEV; usa la altura del sensor
#                            calibrada para distancia metros, y número
#                            de frames consecutivos para confianza.
#
# Hoy la detección está embebida en `threadLineFollowing.py`. Se traslada
# acá cuando el pipeline nuevo de Fase 4 lo absorba; este módulo es el
# único que decide "hay stopline a X metros" — el BehaviorPlanner (Fase
# 4) consume esa observación para activar Crosswalk/Intersection.
