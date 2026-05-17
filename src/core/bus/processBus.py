"""Bus broker process.

Owns the single ZeroMQ XSUB/XPUB proxy that replaces the legacy
``processGateway`` + multiprocessing queues. Hosted in its own OS process
so its event loop (which is a libzmq C thread that doesn't hold the GIL)
doesn't contend with Python workers.

Lifecycle:
    1. Parent process constructs ``processBus(...)`` and calls ``start()``
       (inherited from ``multiprocessing.Process``).
    2. Child runs ``_init_threads()`` → builds the ``BrokerThread`` with
       binds resolved from env vars / defaults. The ZMQ context is created
       in the child here, never inherited from the parent — that's the
       spawn-mode safety rule.
    3. ``WorkerProcess.run()`` starts the thread and blocks on
       ``self._blocker``.
    4. ``stop()`` sets the blocker; ``stop_threads()`` calls
       ``BrokerThread.stop()`` which closes the XSUB socket → ``zmq.proxy``
       returns → thread exits.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.bus.broker import BrokerThread
from src.core.bus.endpoints import xpub_binds, xsub_binds
from src.templates.workerprocess import WorkerProcess


class processBus(WorkerProcess):
    """Hosts the XSUB/XPUB proxy in its own OS process.

    The ``queueList`` argument is accepted for signature compatibility with
    the rest of the ``WorkerProcess`` zoo. It is intentionally unused — the
    bus does not consume ``multiprocessing`` queues; everything flows
    through ZMQ.
    """

    def __init__(
        self,
        queueList=None,
        logger: Optional[logging.Logger] = None,
        debugging: bool = False,
        ready_event=None,
    ) -> None:
        super().__init__(queueList or {}, ready_event=ready_event)
        self.logger = logger or logging.getLogger("bus.broker")
        self.debugging = debugging

    def _init_threads(self) -> None:
        # Endpoint resolution happens here (in the child) so env-var
        # overrides set just before ``.start()`` still take effect.
        # Both bind functions accept lists, enabling dual ipc://+tcp://
        # binds for the same broker socket.
        thread = BrokerThread(xsub_bind=xsub_binds(), xpub_bind=xpub_binds())
        self.threads.append(thread)
