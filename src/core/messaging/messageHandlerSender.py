"""Legacy import path — kept as a thin re-export from the new bus shim.

The real implementation lives in :mod:`src.core.bus.shim`. Existing call
sites import :class:`messageHandlerSender` from this module; the alias
preserves their imports without any code change while routing every
publish through ZeroMQ.

Sprint 5 (cleanup): once every direct caller migrates to
``from src.core.bus import Node`` we can delete this file.
"""

from src.core.bus.shim import messageHandlerSender  # noqa: F401

__all__ = ["messageHandlerSender"]
