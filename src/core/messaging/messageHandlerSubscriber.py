"""Legacy import path — kept as a thin re-export from the new bus shim.

Mirrors :mod:`src.core.messaging.messageHandlerSender`. See that module
for the migration story; this file exists solely so existing imports of
``messageHandlerSubscriber`` continue to resolve without touching every
call site.
"""

from src.core.bus.shim import messageHandlerSubscriber  # noqa: F401

__all__ = ["messageHandlerSubscriber"]
