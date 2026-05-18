"""Persistencia local del brain — SQLite light, sin migraciones complejas.

Plan E2/E4: separamos calibration (estable entre runs) de sessions
(historial de runs) en DBs distintas para que un crash en una no
contamine la otra.

Acceso vía clases ``CalibrationDB`` y ``SessionsDB`` que encapsulan
una conexión thread-safe (WAL mode + check_same_thread=False).
"""
