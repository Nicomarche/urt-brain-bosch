"""URT Brain Bosch — desktop GUI (PyQt5).

Single-window dashboard that replaces the Angular frontend (now in
``legacy/dashboard-frontend/``). Talks to ``processDashboard.py:5005`` over
SocketIO so it works the same locally (host mode) and against a remote car
(monitor mode).

Entry point: ``python3 -m src.dashboard.gui.main`` (see ``main.py``). The
launcher ``run.sh`` is the canonical way to start it.
"""
