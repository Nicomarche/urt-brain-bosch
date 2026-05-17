"""Qt widgets for the URT dashboard.

Each widget is a thin view: it takes a ``SocketIOClient`` (the bus) plus its
own state, connects to the relevant pyqtSignals, and emits commands back
through ``client.emit_message(name, value)``. Keeping the widgets dumb means
we can run the GUI in monitor mode without touching the runtime.
"""
