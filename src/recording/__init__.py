"""Persistencia exhaustiva de runs en modo MANUAL.

El productor central es :class:`ManualSessionRecorder` (proceso aparte
arrancado por :mod:`main`). Suscribe a ``STATE_CHANGE`` y, mientras dura
cada entrada a ``SystemMode.MANUAL``, drena todo el bus a JSONL bajo
``$URT_LOG_RUN_DIR/manual_<ts>/``. Para coordinarse con los grabadores
de video que viven dentro de ``processCamera`` (que tap'ean buffers
intra-proceso sin pasar por el bus), publica el topic LATCHED
``MANUAL_RECORDING_SESSION`` con el path absoluto de la subcarpeta.
"""

from src.recording.manual_session_recorder import ManualSessionRecorder

__all__ = ["ManualSessionRecorder"]
