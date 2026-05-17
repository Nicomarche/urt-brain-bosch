"""Legacy import path — kept as a thin alias for :class:`ZmqBusClient`.

Every widget historically imported ``SocketIOClient`` from this module
for type hints. Replacing the underlying transport with ZeroMQ pub/sub
+ UDP video did NOT change the signal/method surface those widgets
depend on, so we alias the new class under the old name.

Sprint-N cleanup: when every widget is updated to ``from ..client.zmq_bus_client
import ZmqBusClient`` we can delete this file.
"""

from src.gui.client.zmq_bus_client import ZmqBusClient as SocketIOClient  # noqa: F401

__all__ = ["SocketIOClient"]
