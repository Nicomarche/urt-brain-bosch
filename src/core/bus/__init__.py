"""Public API of the pub/sub bus.

Import from this package, not from the internal modules::

    from src.core.bus import Node, Publisher, Subscriber, Topic, qos

The internal modules (broker, processBus, shim, serialization, endpoints)
are implementation details — they may move or be renamed in later sprints.
"""

from src.core.bus import qos
from src.core.bus.node import Node
from src.core.bus.publisher import Publisher
from src.core.bus.qos import QoSProfile
from src.core.bus.subscriber import Subscriber
from src.core.bus.topics import Topic

__all__ = [
    "Node",
    "Publisher",
    "Subscriber",
    "Topic",
    "QoSProfile",
    "qos",
]
