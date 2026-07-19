"""Centralized logging setup for InfernoOps.

Call ``configure_logging()`` once from each entrypoint (app.py, mcp_server.py,
scripts). Library modules should only call ``logging.getLogger(__name__)`` and
never call ``print`` or configure handlers themselves.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once for the whole application.

    Idempotent: calling this more than once will not add duplicate handlers.

    Args:
        level: Minimum log level to emit.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
